import 'dart:convert';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:http/http.dart' as http;

import '../models/project.dart';
import '../models/investor.dart';

// ══════════════════════════════════════════════════════════
// ⚙️  غيّر remoteUrl بعد النشر على Render
// ══════════════════════════════════════════════════════════
class MLConstants {
  static const String localUrl  = 'http://10.0.2.2:8000'; // محاكي Android
  static const String remoteUrl = 'https://YOUR-APP.onrender.com'; // بعد النشر

  static const String baseUrl   = remoteUrl; // ← غيّر هذا عند النشر
  static const Duration timeout = Duration(seconds: 15);
}

// ── Provider ─────────────────────────────────────────────
final mlServiceProvider = Provider<MLService>((ref) => MLService());

// ══════════════════════════════════════════════════════════
class MLService {
  final _client = http.Client();

  // ── تنبؤ: مستثمر واحد ↔ مشروع واحد ──────────────────
  Future<MLPrediction> predict({
    required Project  project,
    required Investor investor,
  }) async {
    final uri  = Uri.parse('${MLConstants.baseUrl}/predict');

    // ✅ نرسل نص الوصف الكامل — هذا ما يحتاجه النموذج
    final body = jsonEncode({
      'project_description': project.description,         // ← الأهم
      'project_title':       project.title,
      'project_category':    project.industry,
      'funding_goal':        project.fundingGoal,
      'investor_name':       investor.name,
      'investor_bio':        investor.bio ?? '',
      'investor_industries': investor.criteria?.industries ?? [],
    });

    try {
      final res = await _client
          .post(uri, headers: {'Content-Type': 'application/json'}, body: body)
          .timeout(MLConstants.timeout);

      if (res.statusCode == 200) {
        return MLPrediction.fromJson(jsonDecode(res.body));
      }
      throw Exception('API ${res.statusCode}');
    } catch (_) {
      return _localFallback(project, investor);
    }
  }

  // ── تنبؤ: مشروع واحد ↔ عدة مستثمرين ────────────────
  Future<List<MLBulkResult>> predictBulk({
    required Project        project,
    required List<Investor> investors,
  }) async {
    if (investors.isEmpty) return [];

    final uri  = Uri.parse('${MLConstants.baseUrl}/predict/bulk');
    final body = jsonEncode({
      'project_description': project.description,
      'project_title':       project.title,
      'project_category':    project.industry,
      'funding_goal':        project.fundingGoal,
      'investors': investors.map((inv) => {
        'id':         inv.id,
        'name':       inv.name,
        'bio':        inv.bio ?? '',
        'industries': inv.criteria?.industries ?? [],
      }).toList(),
    });

    try {
      final res = await _client
          .post(uri, headers: {'Content-Type': 'application/json'}, body: body)
          .timeout(MLConstants.timeout);

      if (res.statusCode == 200) {
        final List data = jsonDecode(res.body);
        return data.map((j) => MLBulkResult.fromJson(j)).toList();
      }
      throw Exception('API ${res.statusCode}');
    } catch (_) {
      return investors.map((inv) {
        final r = _localFallback(project, inv);
        return MLBulkResult(
          investorId:      inv.id,
          investorName:    inv.name,
          decision:        r.decision,
          probability:     r.probability,
          matchPercentage: r.matchPercentage,
          confidenceLevel: r.confidenceLevel,
          positiveSignals: r.positiveSignals,
          negativeSignals: r.negativeSignals,
        );
      }).toList()
        ..sort((a, b) => b.probability.compareTo(a.probability));
    }
  }

  // ── Fallback محلي ────────────────────────────────────
  // يُستخدم إذا كان السيرفر غير متاح
  MLPrediction _localFallback(Project project, Investor investor) {
    final c          = investor.criteria;
    double score     = 0;
    final pos        = <String>[];
    final neg        = <String>[];
    final desc_lower = project.description.toLowerCase();

    // فحص الكلمات الإيجابية
    const positiveKw = [
      'revenue', 'patent', 'retention', 'subscribers',
      'growth', 'margin', 'profit', 'traction'
    ];
    const negativeKw = [
      'pre-revenue', 'crowded', 'saturated', 'no patent'
    ];
    for (final kw in positiveKw) {
      if (desc_lower.contains(kw)) pos.add(kw);
    }
    for (final kw in negativeKw) {
      if (desc_lower.contains(kw)) neg.add(kw);
    }

    if (c != null) {
      if (c.industries.contains(project.industry)) score += 0.40;
      if (c.stages.contains(project.stage))         score += 0.30;
      if (c.minInvestment <= project.fundingGoal &&
          project.fundingGoal <= c.maxInvestment)   score += 0.20;
    }
    score += pos.length * 0.05;
    score -= neg.length * 0.10;
    score  = score.clamp(0.0, 1.0);

    final decision = score >= 0.47 ? 'INVEST' : 'SKIP';
    return MLPrediction(
      decision:        decision,
      probability:     double.parse(score.toStringAsFixed(4)),
      matchPercentage: double.parse((score * 100).toStringAsFixed(1)),
      confidenceLevel: score >= 0.75 || score <= 0.25 ? 'High' : 'Medium',
      positiveSignals: pos,
      negativeSignals: neg,
      explanation:     '$decision — Local fallback (server unavailable)',
    );
  }
}

// ══════════════════════════════════════════════════════════
// Response Models
// ══════════════════════════════════════════════════════════

class MLPrediction {
  final String       decision;         // "INVEST" or "SKIP"
  final double       probability;      // 0.0 → 1.0
  final double       matchPercentage;  // 0 → 100
  final String       confidenceLevel;  // "High" / "Medium" / "Low"
  final List<String> positiveSignals;
  final List<String> negativeSignals;
  final String       explanation;

  MLPrediction({
    required this.decision,
    required this.probability,
    required this.matchPercentage,
    required this.confidenceLevel,
    required this.positiveSignals,
    required this.negativeSignals,
    required this.explanation,
  });

  factory MLPrediction.fromJson(Map<String, dynamic> j) => MLPrediction(
    decision:        j['decision']         as String,
    probability:     (j['probability']     as num).toDouble(),
    matchPercentage: (j['match_percentage'] as num).toDouble(),
    confidenceLevel: j['confidence_level'] as String,
    positiveSignals: List<String>.from(j['positive_signals']),
    negativeSignals: List<String>.from(j['negative_signals']),
    explanation:     j['explanation']      as String,
  );
}

class MLBulkResult {
  final String       investorId;
  final String       investorName;
  final String       decision;
  final double       probability;
  final double       matchPercentage;
  final String       confidenceLevel;
  final List<String> positiveSignals;
  final List<String> negativeSignals;

  MLBulkResult({
    required this.investorId,
    required this.investorName,
    required this.decision,
    required this.probability,
    required this.matchPercentage,
    required this.confidenceLevel,
    required this.positiveSignals,
    required this.negativeSignals,
  });

  factory MLBulkResult.fromJson(Map<String, dynamic> j) => MLBulkResult(
    investorId:      j['investor_id']      as String,
    investorName:    j['investor_name']    as String,
    decision:        j['decision']         as String,
    probability:     (j['probability']     as num).toDouble(),
    matchPercentage: (j['match_percentage'] as num).toDouble(),
    confidenceLevel: j['confidence_level'] as String,
    positiveSignals: List<String>.from(j['positive_signals']),
    negativeSignals: List<String>.from(j['negative_signals']),
  );
}
