// ✅ match_provider.dart — مرتبط بنموذج ML الحقيقي
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../core/supabase/supabase_service.dart';
import '../models/match.dart';
import 'ml_service.dart';

class MatchesState {
  final List<Match> matchedInvestors;
  final List<Match> matchedProjects;
  final bool        isLoading;
  final String?     error;

  const MatchesState({
    this.matchedInvestors = const [],
    this.matchedProjects  = const [],
    this.isLoading        = false,
    this.error,
  });

  MatchesState copyWith({
    List<Match>? matchedInvestors,
    List<Match>? matchedProjects,
    bool?        isLoading,
    String?      error,
  }) => MatchesState(
    matchedInvestors: matchedInvestors ?? this.matchedInvestors,
    matchedProjects:  matchedProjects  ?? this.matchedProjects,
    isLoading:        isLoading        ?? this.isLoading,
    error:            error,
  );
}

class MatchesNotifier extends StateNotifier<MatchesState> {
  final SupabaseService _supabase;
  final MLService       _ml;

  MatchesNotifier(this._supabase, this._ml) : super(const MatchesState());

  // ── مستثمرون مناسبون لمشاريع رائد الأعمال ───────────
  Future<void> fetchMatchedInvestors() async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final myProjects = await _supabase.getMyProjects();
      if (myProjects.isEmpty) {
        state = state.copyWith(matchedInvestors: [], isLoading: false);
        return;
      }

      final investors = await _supabase.getInvestors(limit: 50);
      if (investors.isEmpty) {
        state = state.copyWith(matchedInvestors: [], isLoading: false);
        return;
      }

      final project = myProjects.first;

      // ✅ استدعاء النموذج — يرسل نص الوصف الكامل
      final results = await _ml.predictBulk(
        project:   project,
        investors: investors,
      );

      final matches = results
          .where((r) => r.decision == 'INVEST')
          .map((r) {
            final investor = investors.firstWhere(
              (inv) => inv.id == r.investorId,
              orElse: () => investors.first,
            );
            return Match(
              id:               'ml_${r.investorId}',
              targetId:         r.investorId,
              targetType:       'investor',
              matchPercentage:  r.matchPercentage,
              matchingCriteria: [
                r.confidenceLevel == 'High'
                    ? '✅ High confidence match'
                    : '📊 ${r.matchPercentage.toStringAsFixed(0)}% match',
                if (r.positiveSignals.isNotEmpty)
                  '🟢 ${r.positiveSignals.take(2).join(', ')}',
                if (r.negativeSignals.isNotEmpty)
                  '🔴 ${r.negativeSignals.take(1).join(', ')}',
              ],
              investor: investor,
            );
          })
          .toList();

      state = state.copyWith(matchedInvestors: matches, isLoading: false);
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }

  // ── مشاريع مناسبة للمستثمر ───────────────────────────
  Future<void> fetchMatchedProjects() async {
    state = state.copyWith(isLoading: true, error: null);
    try {
      final uid = _supabase.client.auth.currentUser?.id;
      if (uid == null) { state = state.copyWith(isLoading: false); return; }

      final allInvestors = await _supabase.getInvestors(limit: 100);
      final myInvestor   = allInvestors.where((i) => i.userId == uid).firstOrNull;
      if (myInvestor == null) {
        state = state.copyWith(matchedProjects: [], isLoading: false);
        return;
      }

      final projects = await _supabase.getProjects(limit: 50);
      final List<Match> matches = [];

      for (final project in projects) {
        final result = await _ml.predict(
          project:  project,
          investor: myInvestor,
        );

        if (result.decision == 'INVEST') {
          matches.add(Match(
            id:               'ml_${project.id}',
            targetId:         project.id,
            targetType:       'project',
            matchPercentage:  result.matchPercentage,
            matchingCriteria: [
              '${result.matchPercentage.toStringAsFixed(0)}% match — ${result.confidenceLevel} confidence',
              if (result.positiveSignals.isNotEmpty)
                'Signals: ${result.positiveSignals.take(3).join(', ')}',
            ],
            project: project,
          ));
        }
      }

      matches.sort((a, b) =>
          b.matchPercentage.compareTo(a.matchPercentage));

      state = state.copyWith(matchedProjects: matches, isLoading: false);
    } catch (e) {
      state = state.copyWith(isLoading: false, error: e.toString());
    }
  }

  void clearError() => state = state.copyWith(error: null);
}

final matchesProvider =
    StateNotifierProvider<MatchesNotifier, MatchesState>((ref) {
  return MatchesNotifier(
    ref.watch(supabaseServiceProvider),
    ref.watch(mlServiceProvider),
  );
});
