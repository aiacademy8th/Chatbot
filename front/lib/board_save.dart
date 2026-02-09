import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http; // http 패키지 추가

class BoardSaveScreen extends StatefulWidget {
  // UI에 보여줄 데이터
  final String analysisContent;
  final int postId;

  // ⭐ [기능 추가] 백엔드 전송용 숨겨진 데이터 (UI엔 영향 없음)
  final String faultRatio;
  final String legalBasis;
  final String accidentInfo;

  const BoardSaveScreen({
    super.key,
    required this.analysisContent,
    required this.postId,
    // 생성자에서 데이터 받기
    required this.faultRatio,
    required this.legalBasis,
    required this.accidentInfo,
  });

  @override
  State<BoardSaveScreen> createState() => _BoardSaveScreenState();
}

class _BoardSaveScreenState extends State<BoardSaveScreen> {
  final TextEditingController _passwordController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _passwordController.dispose();
    super.dispose();
  }

  // ⭐ [기능 수정] 실제 서버 저장 로직 구현
  Future<void> _saveToBoard() async {
    // 1. 비밀번호 입력 확인
    if (_passwordController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('비밀번호를 입력해주세요')),
      );
      return;
    }

    // 2. 비밀번호 길이 확인 (백엔드 제약: 4자리 숫자)
    if (_passwordController.text.length != 4) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('비밀번호는 4자리 숫자여야 합니다')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      // 3. 백엔드 URL 설정
      final url = Uri.parse('https://chatbot-backend-599050237852.asia-northeast3.run.app/board');

      // 4. 제목 자동 생성 (UI에 입력창이 없으므로 자동 생성)
      final String autoTitle = '${DateTime.now().toString().substring(0, 10)} 교통사고 분석 리포트';

      // 5. 데이터 패키징 (백엔드 규격 준수)
      final Map<String, dynamic> requestBody = {
        "board_password": _passwordController.text,
        "accident_title": autoTitle,              // 자동 생성 제목
        "accident_info": widget.accidentInfo,     // 넘겨받은 사고 정황
        "fault_ratio": widget.faultRatio,         // 넘겨받은 과실 비율
        "analysis_result": widget.analysisContent,// 넘겨받은 분석 결과
        "legal_basis": widget.legalBasis,         // 넘겨받은 법적 근거
        "accident_summary": widget.analysisContent.length > 50 
            ? "${widget.analysisContent.substring(0, 50)}..." 
            : widget.analysisContent
      };

      // 6. 전송
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(requestBody),
      );

      if (response.statusCode == 200) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('게시글이 저장되었습니다')),
          );

          await Future.delayed(const Duration(seconds: 1));
          if (mounted) {
            Navigator.pop(context);
          }
        }
      } else {
        throw Exception('서버 오류: ${response.statusCode}');
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('저장 실패: $e')),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  // --- 아래 UI 코드는 업로드해주신 원본과 100% 동일합니다 ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('게시판 저장'),
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 게시글 ID
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.blue.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.blue.shade200),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '📌 게시글 ID',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(color: Colors.grey.shade300),
                      ),
                      child: Text(
                        '${widget.postId}',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.blue,
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      '(자동 생성된 고유 번호입니다)',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey.shade600,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 32),

              // 비밀번호 입력
              const Text(
                '🔐 비밀번호',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              TextField(
                controller: _passwordController,
                obscureText: true,
                decoration: InputDecoration(
                  hintText: '비밀번호를 입력해주세요',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                  ),
                  focusedBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(8),
                    borderSide: const BorderSide(
                      color: Colors.blue,
                      width: 2,
                    ),
                  ),
                  contentPadding: const EdgeInsets.all(16),
                ),
              ),

              const SizedBox(height: 32),

              // 게시 내용 미리보기
              const Text(
                '📄 게시 내용',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.grey.shade100,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: Text(
                  widget.analysisContent,
                  style: const TextStyle(
                    fontSize: 13,
                    height: 1.6,
                  ),
                  maxLines: 8,
                  overflow: TextOverflow.ellipsis,
                ),
              ),

              const SizedBox(height: 40),

              // 저장 버튼
              SizedBox(
                width: double.infinity,
                height: 56,
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _saveToBoard,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green.shade600,
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    elevation: 4,
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          width: 24,
                          height: 24,
                          child: CircularProgressIndicator(
                            valueColor:
                                AlwaysStoppedAnimation<Color>(Colors.white),
                            strokeWidth: 3,
                          ),
                        )
                      : const Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.check_circle, size: 24),
                            SizedBox(width: 8),
                            Text(
                              '게시글 저장',
                              style: TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}