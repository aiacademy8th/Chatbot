import 'package:flutter/material.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'board_save.dart';
import 'chat.dart';

class ResultScreen extends StatelessWidget {
  final Map<String, dynamic> analysisResult;
  final Map<String, dynamic> userAnswers;
  final String threadId;

  const ResultScreen({
    super.key,
    required this.analysisResult,
    required this.userAnswers,
    required this.threadId,
  });

  int _generatePostId() {
    return DateTime.now().microsecond % 1000000 + 1;
  }

  int _extractFaultPercentage(Map<String, dynamic> ratioMap) {
    return ratioMap['opponent'] as int? ?? 50;
  }

  String _getRecommendedAction(int opponentFault) {
    if (opponentFault >= 80) return '보험 유리';
    if (opponentFault >= 50) return '보험 권장';
    return '합의 유리';
  }

  String _getActionEmoji(String action) {
    switch (action) {
      case '보험 유리':
        return '🛡️';
      case '보험 권장':
        return '⚖️';
      case '합의 유리':
        return '🤝';
      default:
        return '⚪';
    }
  }

  List<Color> _getActionGradient(String action) {
    switch (action) {
      case '보험 유리':
        return [const Color(0xFFEF5350), const Color(0xFFE53935)];
      case '보험 권장':
        return [const Color(0xFFFFB74D), const Color(0xFFFB8C00)];
      case '합의 유리':
        return [const Color(0xFF66BB6A), const Color(0xFF43A047)];
      default:
        return [Colors.grey, Colors.grey];
    }
  }

  Color _getActionColor(String action) {
    switch (action) {
      case '보험 유리':
        return const Color(0xFFE53935);
      case '보험 권장':
        return const Color(0xFFFB8C00);
      case '합의 유리':
        return const Color(0xFF43A047);
      default:
        return Colors.grey;
    }
  }

  Future<void> _generatePDF(BuildContext context) async {
    try {
      final fontData = await rootBundle.load('assets/fonts/NotoSansKR-Regular.ttf');
      final ttf = pw.Font.ttf(fontData);
      final pdfTheme = pw.ThemeData.withFont(base: ttf, bold: ttf);

      final pdf = pw.Document(theme: pdfTheme);

      pdf.addPage(
        pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(32),
          build: (pw.Context context) {
            return [
              pw.Header(
                level: 0,
                child: pw.Text(
                  '교통사고 과실 비율 분석 결과',
                  style: pw.TextStyle(fontSize: 24, fontWeight: pw.FontWeight.bold),
                ),
              ),
              pw.SizedBox(height: 20),
              pw.Text(
                '분석 일시: ${DateTime.now().toString().substring(0, 19)}',
                style: const pw.TextStyle(fontSize: 12, color: PdfColors.grey700),
              ),
              pw.SizedBox(height: 30),
              pw.Header(level: 1, text: '사고 정보'),
              pw.SizedBox(height: 10),
              ...userAnswers.entries.map((entry) {
                return pw.Padding(
                  padding: const pw.EdgeInsets.only(bottom: 8),
                  child: pw.Row(
                    crossAxisAlignment: pw.CrossAxisAlignment.start,
                    children: [
                      pw.SizedBox(
                        width: 150,
                        child: pw.Text('${entry.key}:',
                            style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
                      ),
                      pw.Expanded(child: pw.Text('${entry.value}')),
                    ],
                  ),
                );
              }).toList(),
              pw.SizedBox(height: 30),
              pw.Header(level: 1, text: '분석 결과'),
              pw.SizedBox(height: 10),
              pw.Text(
                analysisResult['result']?['reasoning'] ?? analysisResult['analysis'] ?? '분석 결과 없음',
                style: const pw.TextStyle(fontSize: 12, lineSpacing: 1.5),
              ),
            ];
          },
        ),
      );

      await Printing.sharePdf(bytes: await pdf.save(), filename: 'accident_analysis.pdf');
    } catch (e) {
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('PDF 저장 실패: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final Map<String, dynamic> actualAnalysisData =
        analysisResult['result'] as Map<String, dynamic>? ?? {};

    final faultRatio = actualAnalysisData['fault_ratio'] as Map<String, dynamic>? ??
        {'me': 50, 'opponent': 50};
    final myFault = faultRatio['me'] as int? ?? 50;
    final opponentFault = _extractFaultPercentage(faultRatio);
    final recommendedAction = _getRecommendedAction(opponentFault);
    final actionEmoji = _getActionEmoji(recommendedAction);
    final actionGradient = _getActionGradient(recommendedAction);
    final actionColor = _getActionColor(recommendedAction);

    final analysisText = actualAnalysisData['reasoning'] ?? actualAnalysisData['analysis'] ?? '분석 결과가 없습니다.';
    final referencesList = actualAnalysisData['legal_basis'] as List? ?? [];

    return Scaffold(
      backgroundColor: const Color(0xFFF7F8F5),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: Padding(
          padding: const EdgeInsets.all(8),
          child: GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              decoration: const BoxDecoration(
                color: Color(0xFFF0F0F0),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_back, size: 18, color: Color(0xFF555555)),
            ),
          ),
        ),
        title: const Text(
          '분석 결과',
          style: TextStyle(
            fontSize: 22,
            fontWeight: FontWeight.w700,
            color: Color(0xFF222222),
          ),
        ),
        centerTitle: true,
        actions: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: GestureDetector(
              onTap: () => _generatePDF(context),
              child: Container(
                width: 34,
                height: 34,
                decoration: const BoxDecoration(
                  color: Color(0xFFF0F0F0),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.share, size: 18, color: Color(0xFF555555)),
              ),
            ),
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFF0F0F0)),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 20),

            // ===== 큰 이모지 뱃지 =====
            Center(
              child: Text(
                actionEmoji,
                style: const TextStyle(fontSize: 60),
              ),
            ),
            const SizedBox(height: 8),
            Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
                decoration: BoxDecoration(
                  gradient: LinearGradient(colors: actionGradient),
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: actionColor.withOpacity(0.3),
                      blurRadius: 12,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Text(
                  recommendedAction,
                  style: const TextStyle(
                    fontSize: 21,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
            ),

            const SizedBox(height: 20),

            // ===== 과실 비율 카드 (바 차트) =====
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 12,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Column(
                  children: [
                    const Text(
                      '📊 과실 비율',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF333333),
                      ),
                    ),
                    const SizedBox(height: 16),

                    // 라벨
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '🚗 나',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: const Color(0xFF1E88E5),
                          ),
                        ),
                        Text(
                          '🚙 상대',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: const Color(0xFFE53935),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),

                    // 바 차트
                    ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: SizedBox(
                        height: 32,
                        child: Row(
                          children: [
                            Expanded(
                              flex: myFault,
                              child: Container(
                                decoration: const BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [Color(0xFF42A5F5), Color(0xFF1E88E5)],
                                  ),
                                ),
                              ),
                            ),
                            Expanded(
                              flex: opponentFault,
                              child: Container(
                                decoration: const BoxDecoration(
                                  gradient: LinearGradient(
                                    colors: [Color(0xFFEF5350), Color(0xFFE53935)],
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 10),

                    // 숫자
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '$myFault',
                          style: const TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.w900,
                            color: Color(0xFF1E88E5),
                          ),
                        ),
                        Text(
                          '$opponentFault',
                          style: const TextStyle(
                            fontSize: 32,
                            fontWeight: FontWeight.w900,
                            color: Color(0xFFE53935),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 14),

            // ===== 상세 분석 카드 =====
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.03),
                      blurRadius: 8,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '📝 상세 분석',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF333333),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      analysisText,
                      style: const TextStyle(
                        fontSize: 15,
                        height: 1.7,
                        color: Color(0xFF555555),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 14),

            // ===== 참고 자료 =====
            if (referencesList.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(18),
                    border: Border.all(color: const Color(0xFFF0F0F0)),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.03),
                        blurRadius: 8,
                        offset: const Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        '📚 참고 자료',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF333333),
                        ),
                      ),
                      const SizedBox(height: 12),
                      ...referencesList.map((ref) {
                        final referenceMap = ref is Map<String, dynamic>
                            ? ref
                            : {'source': ref.toString(), 'content': ''};
                        return Container(
                          margin: const EdgeInsets.only(bottom: 10),
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: const Color(0xFFFFFDE7),
                            borderRadius: BorderRadius.circular(12),
                            border: Border.all(color: const Color(0xFFFFF9C4)),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  const Text('📄', style: TextStyle(fontSize: 14)),
                                  const SizedBox(width: 6),
                                  Expanded(
                                    child: Text(
                                      referenceMap['source'] ?? '출처 미상',
                                      style: const TextStyle(
                                        fontWeight: FontWeight.w700,
                                        fontSize: 14,
                                        color: Color(0xFF555555),
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                              if ((referenceMap['content'] ?? '').isNotEmpty) ...[
                                const SizedBox(height: 6),
                                Text(
                                  referenceMap['content'] ?? '',
                                  style: const TextStyle(
                                    fontSize: 13,
                                    color: Color(0xFF888888),
                                    height: 1.5,
                                  ),
                                ),
                              ],
                            ],
                          ),
                        );
                      }).toList(),
                    ],
                  ),
                ),
              ),

            const SizedBox(height: 14),

            // ===== 하단 버튼 3개 =====
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                children: [
                  // PDF 저장
                  Expanded(
                    child: GestureDetector(
                      onTap: () => _generatePDF(context),
                      child: Container(
                        height: 54,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: const Color(0xFFFFCDD2), width: 1.5),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Text('📑', style: TextStyle(fontSize: 17)),
                            SizedBox(width: 5),
                            Text(
                              'PDF',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFFE53935),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),

                  // 추가 질문
                  Expanded(
                    child: GestureDetector(
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ChatbotScreen(
                              accidentPhotos: const [],
                              threadId: threadId,
                              initialChatMode: true,
                            ),
                          ),
                        );
                      },
                      child: Container(
                        height: 54,
                        decoration: BoxDecoration(
                          gradient: const LinearGradient(
                            colors: [Color(0xFF66BB6A), Color(0xFF43A047)],
                          ),
                          borderRadius: BorderRadius.circular(14),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0xFF43A047).withOpacity(0.25),
                              blurRadius: 8,
                              offset: const Offset(0, 3),
                            ),
                          ],
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Text('💬', style: TextStyle(fontSize: 17)),
                            SizedBox(width: 5),
                            Text(
                              '추가 질문',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),

                  // ⭐ [수정된 부분] 게시판 저장
                  Expanded(
                    child: GestureDetector(
                      onTap: () async {
                        final text = analysisText;
                        Clipboard.setData(ClipboardData(text: text));

                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('분석 내용이 복사되었습니다'),
                            duration: Duration(seconds: 2),
                          ),
                        );

                        await Future.delayed(const Duration(seconds: 1));

                        if (context.mounted) {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => BoardSaveScreen(
                                analysisContent: text,
                                postId: _generatePostId(),
                                // ▼▼▼ [중요] 개별 파라미터가 아닌 Map 통째로 전달
                                fullResult: analysisResult, 
                                userAnswers: userAnswers,
                              ),
                            ),
                          );
                        }
                      },
                      child: Container(
                        height: 54,
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(color: const Color(0xFFBBDEFB), width: 1.5),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: const [
                            Text('📋', style: TextStyle(fontSize: 17)),
                            SizedBox(width: 5),
                            Text(
                              '게시판',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF1E88E5),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 12),

            // ===== 주의사항 =====
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: const [
                    Text('⚠️', style: TextStyle(fontSize: 13)),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        '본 분석 결과는 AI 기반 예측이며, 실제 보험사 또는 법원의 판단과 다를 수 있습니다. 정확한 과실 비율 판정을 위해서는 전문가와 상담하시기 바랍니다.',
                        style: TextStyle(
                          fontSize: 13,
                          color: Color(0xFFAAAAAA),
                          height: 1.5,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}