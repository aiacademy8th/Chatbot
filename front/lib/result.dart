import 'package:flutter/material.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'board_save.dart'; 
import 'chat.dart'; // ChatWidgetScreen -> ChatbotScreen으로 변경

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

  // ⭐ 게시글 ID 생성 (1부터 시작)
  int _generatePostId() {
    return DateTime.now().microsecond % 1000000 + 1;
  }

  // 과실 비율 Map에서 상대방 과실 숫자 추출
  int _extractFaultPercentage(Map<String, dynamic> ratioMap) {
    // "opponent" 키의 값을 정수로 반환, 없으면 50
    return ratioMap['opponent'] as int? ?? 50;
  }

  // 추천 액션 결정
  String _getRecommendedAction(int opponentFault) {
    if (opponentFault >= 80) {
      return '보험 유리';
    } else if (opponentFault >= 50) {
      return '보험 권장';
    } else {
      return '합의 유리';
    }
  }

  // 액션별 색상
  Color _getActionColor(String action) {
    switch (action) {
      case '보험 유리':
        return Colors.red;
      case '보험 권장':
        return Colors.orange;
      case '합의 유리':
        return Colors.green;
      default:
        return Colors.grey;
    }
  }

  // PDF 생성 및 저장
  Future<void> _generatePDF(BuildContext context) async {
    final fontData = await rootBundle.load('assets/fonts/NotoSansKR-Regular.ttf');
    final ttf = pw.Font.ttf(fontData);
    final pdfTheme = pw.ThemeData.withFont(base: ttf);
    
    final pdf = pw.Document(theme: pdfTheme);

    pdf.addPage(
      pw.MultiPage(
        pageFormat: PdfPageFormat.a4,
        margin: const pw.EdgeInsets.all(32),
        build: (pw.Context context) {
          return [
            // 제목
            pw.Header(
              level: 0,
              child: pw.Text(
                '교통사고 과실 비율 분석 결과',
                style: pw.TextStyle(
                  fontSize: 24, 
                  fontWeight: pw.FontWeight.bold,
                ),
              ),
            ),
            pw.SizedBox(height: 20),

            // 분석 일시
            pw.Text(
              '분석 일시: ${DateTime.now().toString().substring(0, 19)}',
              style: const pw.TextStyle(fontSize: 12, color: PdfColors.grey700),
            ),
            pw.SizedBox(height: 30),

            // 사고 정보
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
                    pw.Expanded(
                      child: pw.Text('${entry.value}'),
                    ),
                  ],
                ),
              );
            }).toList(),
            pw.SizedBox(height: 30),

            // 분석 결과
            pw.Header(level: 1, text: '분석 결과'),
            pw.SizedBox(height: 10),
            pw.Text(
              analysisResult['analysis'] ?? '분석 결과가 없습니다.',
              style: const pw.TextStyle(fontSize: 12, lineSpacing: 1.5),
            ),
            pw.SizedBox(height: 30),

            // 참고 자료 (RAG 결과)
            if (analysisResult['references'] != null) ...[
              pw.Header(level: 1, text: '참고 자료'),
              pw.SizedBox(height: 10),
              ...((analysisResult['references'] as List).map((ref) {
                return pw.Padding(
                  padding: const pw.EdgeInsets.only(bottom: 12),
                  child: pw.Column(
                    crossAxisAlignment: pw.CrossAxisAlignment.start,
                    children: [
                      pw.Text('• ${ref['source']}',
                        style: pw.TextStyle(fontWeight: pw.FontWeight.bold)),
                      pw.Padding(
                        padding: const pw.EdgeInsets.only(left: 12, top: 4),
                        child: pw.Text(ref['content'],
                          style: const pw.TextStyle(fontSize: 11)),
                      ),
                    ],
                  ),
                );
              }).toList()),
            ],

            pw.SizedBox(height: 30),

            // 주의사항
            pw.Container(
              padding: const pw.EdgeInsets.all(16),
              decoration: pw.BoxDecoration(
                border: pw.Border.all(color: PdfColors.grey400),
                borderRadius: const pw.BorderRadius.all(pw.Radius.circular(8)),
              ),
              child: pw.Column(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                children: [
                  pw.Text('⚠️ 주의사항',
                    style: pw.TextStyle(fontSize: 14, fontWeight: pw.FontWeight.bold)),
                  pw.SizedBox(height: 8),
                  pw.Text(
                    '본 분석 결과는 AI 기반 예측이며, 실제 보험사 또는 법원의 판단과 다를 수 있습니다. '
                    '정확한 과실 비율 판정을 위해서는 전문가와 상담하시기 바랍니다.',
                    style: const pw.TextStyle(fontSize: 10),
                  ),
                ],
              ),
            ),
          ];
        },
      ),
    );

    // PDF 저장
    try {
      final output = await getDownloadsDirectory(); // ⭐ 변경
      final fileName = 'accident_analysis_${DateTime.now().millisecondsSinceEpoch}.pdf';
      final file = File('${output!.path}/$fileName');
      await file.writeAsBytes(await pdf.save());
      // final output = await getApplicationDocumentsDirectory();
      // final file = File('${output.path}/accident_analysis_${DateTime.now().millisecondsSinceEpoch}.pdf');
      // await file.writeAsBytes(await pdf.save());

      if (context.mounted) {
        print('📁 PDF 저장 경로: ${file.path}'); 

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('PDF 저장 완료: ${file.path}'),
            duration: const Duration(seconds: 3),
            action: SnackBarAction(
              label: '공유',
              onPressed: () async {
                await Printing.sharePdf(
                  bytes: await pdf.save(),
                  filename: 'accident_analysis.pdf',
                );
              },
            ),
          ),
        );
      }
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
    // analysisResult는 백엔드 응답 전체를 포함하므로, 'result' 키에서 실제 분석 데이터를 추출합니다.
    final Map<String, dynamic> actualAnalysisData = analysisResult['result'] as Map<String, dynamic>? ?? {};

    // 'fault_ratio'를 Map으로 받고, null일 경우 기본값을 설정합니다.
    final faultRatio = actualAnalysisData['fault_ratio'] as Map<String, dynamic>? ?? {'me': 50, 'opponent': 50};
    final opponentFault = _extractFaultPercentage(faultRatio);
    final recommendedAction = _getRecommendedAction(opponentFault);
    final actionColor = _getActionColor(recommendedAction);

    // 상세 분석 내용과 참고 자료를 추출합니다.
    final analysisText = actualAnalysisData['reasoning'] ?? '분석 결과가 없습니다.';
    final referencesList = actualAnalysisData['legal_basis'] as List? ?? [];

    return Scaffold(
      appBar: AppBar(
        title: const Text('분석 결과'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () async {
              final pdf = pw.Document();
              // PDF 생성 로직 (위와 동일)
              await Printing.sharePdf(
                bytes: await pdf.save(),
                filename: 'accident_analysis.pdf',
              );
            },
          ),
        ],
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            // 신호등 표시
            Container(
              padding: const EdgeInsets.symmetric(vertical: 32),
              color: Colors.grey.shade100,
              child: Column(
                children: [
                  const Text(
                    '추천 조치',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                  const SizedBox(height: 24),
                  
                  // 신호등 표시
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _buildTrafficLight(
                        '보험 유리',
                        Colors.red,
                        recommendedAction == '보험 유리'
                        // '80~100%',
                      ),
                      const SizedBox(width: 32),
                      _buildTrafficLight(
                        '보험 권장',
                        Colors.orange,
                        recommendedAction == '보험 권장'
                        // '50~79%',
                      ),
                      const SizedBox(width: 32),
                      _buildTrafficLight(
                        '합의 유리',
                        Colors.green,
                        recommendedAction == '합의 유리'
                        // '~49%',
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  
                  // 선택된 액션 강조
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                    decoration: BoxDecoration(
                      color: actionColor.withOpacity(0.1),
                      borderRadius: BorderRadius.circular(24),
                      border: Border.all(color: actionColor, width: 2),
                    ),
                    child: Text(
                      recommendedAction,
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: actionColor,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // 과실 비율
            Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '📊 예상 과실 비율',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.blue.shade50,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.blue.shade200),
                    ),
                    child: Column(
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceAround,
                          children: [
                            Column(
                              children: [
                                const Text('나', style: TextStyle(fontSize: 16)),
                                const SizedBox(height: 8),
                                Text(
                                  faultRatio['me'].toString(),
                                  style: const TextStyle(
                                    fontSize: 48,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.blue,
                                  ),
                                ),
                                // const Text('%', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                            const Text(
                              ':',
                              style: TextStyle(fontSize: 36, fontWeight: FontWeight.bold),
                            ),
                            Column(
                              children: [
                                const Text('상대', style: TextStyle(fontSize: 16)),
                                const SizedBox(height: 8),
                                Text(
                                  faultRatio['opponent'].toString(),
                                  style: const TextStyle(
                                    fontSize: 48,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.red,
                                  ),
                                ),
                                // const Text('%', style: TextStyle(fontSize: 16)),
                              ],
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // 분석 내용
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '📝 상세 분석',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: Text(
                      analysisText,
                      style: const TextStyle(
                        fontSize: 15,
                        height: 1.6,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 32),

            // 참고 자료 (RAG 결과)
            if (referencesList.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '📚 참고 자료',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ...(referencesList).map((ref) {
                      // ref가 Map<String, dynamic> 타입인지 확인하고 안전하게 사용
                      final referenceMap = ref is Map<String, dynamic> ? ref : {'source': ref.toString(), 'content': ''};
                      return Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: Colors.amber.shade50,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.amber.shade200),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(Icons.article, size: 20, color: Colors.amber),
                                const SizedBox(width: 8),
                                Expanded(
                                  child: Text(
                                    referenceMap['source'] ?? '출처 미상',
                                    style: const TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 14,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Text(
                              referenceMap['content'] ?? '',
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey.shade800,
                                height: 1.4,
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
                  ],
                ),
              ),

            const SizedBox(height: 32),

            // PDF 저장 버튼
            Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                // 첫 번째 버튼
                Expanded(
                  child: SizedBox(
                    height: 56,
                    child: ElevatedButton.icon(
                      onPressed: () => _generatePDF(context),
                      icon: const Icon(Icons.picture_as_pdf, size: 24),
                      label: const Text(
                        'PDF 저장',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.red.shade600,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 4,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),

                // 챗봇 진행 버튼 추가
                Expanded(
                  child: SizedBox(
                    height: 56,
                    child: ElevatedButton.icon(
                      onPressed: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                            builder: (context) => ChatbotScreen(
                              accidentPhotos: const [], // chat.dart에 정의된 ChatbotScreen은 accidentPhotos를 필수로 받음
                              threadId: threadId,
                              initialChatMode: true,
                            ),
                          ),
                        );
                      },
                      icon: const Icon(Icons.chat_bubble_outline, size: 24),
                      label: const Text(
                        '챗봇 진행',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                      ),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green.shade600,
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                        elevation: 4,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                
                // 두 번째 버튼
                Expanded(
                    child: SizedBox(
                      height: 56,
                      child: ElevatedButton.icon(
                        onPressed: () async {
                          // ⭐ 분석 내용 복사
                          final analysisText =
                              analysisResult['analysis'] ??
                              '분석 결과가 없습니다.';
                          Clipboard.setData(
                            ClipboardData(text: analysisText),
                          );

                          // 스낵바 표시
                          ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('분석 내용이 복사되었습니다'),
                              duration: Duration(seconds: 2),
                            ),
                          );

                          // ⭐ 2초 후 게시판 저장 화면으로 이동
                          await Future.delayed(
                            const Duration(seconds: 1),
                          );

                          if (context.mounted) {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (context) => BoardSaveScreen(
                                  analysisContent: analysisText,
                                  postId: _generatePostId(),
                                ),
                              ),
                            );
                          }
                        },
                        icon: const Icon(Icons.save, size: 24),
                        label: const Text(
                          '게시판 저장',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.blue.shade600,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                          elevation: 4,
                        ),
                    ),
                  ),
                ),
              ],
            ),
          ),


            // 주의사항
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 32),
              child: Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.info_outline, color: Colors.grey.shade700, size: 20),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        '본 분석 결과는 AI 기반 예측이며, 실제 보험사 또는 법원의 판단과 다를 수 있습니다. '
                        '정확한 과실 비율 판정을 위해서는 전문가와 상담하시기 바랍니다.',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey.shade700,
                          height: 1.4,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // 신호등 위젯
  Widget _buildTrafficLight(String label, Color color, bool isActive) {
    return Column(
      children: [
        Container(
          width: 80,
          height: 80,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: isActive ? color : Colors.grey.shade300,
            boxShadow: isActive
                ? [
                    BoxShadow(
                      color: color.withOpacity(0.5),
                      blurRadius: 20,
                      spreadRadius: 5,
                    ),
                  ]
                : null,
          ),
          child: isActive
              ? Icon(
                  Icons.check,
                  color: Colors.white,
                  size: 48,
                )
              : null,
        ),
        const SizedBox(height: 12),
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
            color: isActive ? color : Colors.grey.shade600,
          ),
        ),
        const SizedBox(height: 4),
        // Text(
        //   range,
        //   style: TextStyle(
        //     fontSize: 11,
        //     color: Colors.grey.shade600,
        //   ),
        // ),
      ],
    );
  }
}