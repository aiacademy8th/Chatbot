import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class BoardSaveScreen extends StatefulWidget {
  final String analysisContent;
  final int postId;
  final Map<String, dynamic>? fullResult;
  final Map<String, dynamic>? userAnswers;

  const BoardSaveScreen({
    super.key,
    required this.analysisContent,
    required this.postId,
    this.fullResult,
    this.userAnswers,
  });

  @override
  State<BoardSaveScreen> createState() => _BoardSaveScreenState();
}

class _BoardSaveScreenState extends State<BoardSaveScreen> {
  final _titleController = TextEditingController();
  final _passwordController = TextEditingController();
  final _passwordConfirmController = TextEditingController();
  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _obscureConfirm = true;

  final String baseUrl = dotenv.env['BACKEND_URL'] ?? "";

  @override
  void dispose() {
    _titleController.dispose();
    _passwordController.dispose();
    _passwordConfirmController.dispose();
    super.dispose();
  }

  String _buildAccidentInfo() {
    if (widget.userAnswers == null) return '정보 없음';
    final answers = widget.userAnswers!;
    final buffer = StringBuffer();

    if (answers['time_blackbox'] != null) {
      final tb = answers['time_blackbox'];
      if (tb is Map) {
        buffer.writeln('시간: ${tb['time'] ?? '미입력'}, 블랙박스: ${tb['blackbox'] ?? '미입력'}');
      } else {
        buffer.writeln('시간/블랙박스: $tb');
      }
    }
    if (answers['location_type'] != null) buffer.writeln('사고 장소: ${answers['location_type']}');
    if (answers['my_action'] != null) buffer.writeln('나의 주행: ${answers['my_action']}');
    if (answers['opponent_action'] != null) buffer.writeln('상대 주행: ${answers['opponent_action']}');
    if (answers['collision_part'] != null) {
      final parts = answers['collision_part'];
      buffer.writeln('충돌 부위: ${parts is List ? parts.join(', ') : parts}');
    }
    if (answers['fault_factors'] != null) {
      final factors = answers['fault_factors'];
      if (factors is Map) {
        final my = factors['my'] is List ? (factors['my'] as List).join(', ') : '해당없음';
        final op = factors['opponent'] is List ? (factors['opponent'] as List).join(', ') : '해당없음';
        buffer.writeln('나의 과실: $my');
        buffer.writeln('상대 과실: $op');
      } else {
        buffer.writeln('과실 요소: $factors');
      }
    }
    if (answers['additional_info'] != null && answers['additional_info'].toString().isNotEmpty) {
      buffer.writeln('추가 정보: ${answers['additional_info']}');
    }

    return buffer.isEmpty ? '정보 없음' : buffer.toString().trim();
  }

  String _buildFaultRatio() {
    if (widget.fullResult == null) return '정보 없음';
    final result = widget.fullResult!['result'] as Map<String, dynamic>? ?? {};
    final ratio = result['fault_ratio'] as Map<String, dynamic>?;
    if (ratio != null) {
      return '나 ${ratio['me'] ?? '?'} : 상대 ${ratio['opponent'] ?? '?'}';
    }
    return '정보 없음';
  }

  String _buildAnalysisResult() {
    if (widget.fullResult == null) return widget.analysisContent;
    final result = widget.fullResult!['result'] as Map<String, dynamic>? ?? {};
    return result['reasoning'] ?? widget.analysisContent;
  }

  String _buildLegalBasis() {
    if (widget.fullResult == null) return '';
    final result = widget.fullResult!['result'] as Map<String, dynamic>? ?? {};
    final legalBasis = result['legal_basis'] as List? ?? [];
    if (legalBasis.isEmpty) return '참고 자료 없음';

    return legalBasis.map((ref) {
      if (ref is Map<String, dynamic>) {
        return '${ref['source'] ?? ''}: ${ref['content'] ?? ''}';
      }
      return ref.toString();
    }).join('\n');
  }

  String _buildSummary() {
    if (widget.fullResult == null) return widget.analysisContent;
    final result = widget.fullResult!['result'] as Map<String, dynamic>? ?? {};
    return result['summary'] ?? result['reasoning'] ?? widget.analysisContent;
  }

  Future<void> _saveToBoard() async {
    // 유효성 검사
    if (_titleController.text.trim().isEmpty) {
      _showSnackBar('제목을 입력해주세요');
      return;
    }
    if (_passwordController.text.trim().isEmpty) {
      _showSnackBar('비밀번호를 입력해주세요');
      return;
    }
    if (_passwordController.text.length < 4) {
      _showSnackBar('비밀번호는 4자리 이상 입력해주세요');
      return;
    }
    if (_passwordController.text != _passwordConfirmController.text) {
      _showSnackBar('비밀번호가 일치하지 않습니다');
      return;
    }

    setState(() => _isLoading = true);

    try {
      final url = Uri.parse('$baseUrl/board');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'board_password': _passwordController.text.trim(),
          'accident_title': _titleController.text.trim(),
          'accident_info': _buildAccidentInfo(),
          'fault_ratio': _buildFaultRatio(),
          'analysis_result': _buildAnalysisResult(),
          'legal_basis': _buildLegalBasis(),
          'accident_summary': _buildSummary(),
        }),
      );

      if (response.statusCode == 200) {
        if (mounted) {
          _showSuccessDialog();
        }
      } else {
        final errorBody = utf8.decode(response.bodyBytes);
        _showSnackBar('저장 실패: ${response.statusCode}\n$errorBody');
      }
    } catch (e) {
      _showSnackBar('네트워크 오류: $e');
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: const Duration(seconds: 2)),
    );
  }

  void _showSuccessDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (ctx) => Dialog(
        backgroundColor: Colors.white,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        child: Padding(
          padding: const EdgeInsets.all(28),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('🎉', style: TextStyle(fontSize: 48)),
              const SizedBox(height: 12),
              const Text(
                '저장 완료!',
                style: TextStyle(fontSize: 22, fontWeight: FontWeight.w800, color: Color(0xFF222222)),
              ),
              const SizedBox(height: 8),
              const Text(
                '게시판에 성공적으로 저장되었어요',
                style: TextStyle(fontSize: 14, color: Color(0xFFAAAAAA)),
              ),
              const SizedBox(height: 20),
              GestureDetector(
                onTap: () {
                  Navigator.pop(ctx);
                  Navigator.pop(context, true);
                },
                child: Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF66BB6A), Color(0xFF43A047)],
                    ),
                    borderRadius: BorderRadius.circular(14),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFF43A047).withOpacity(0.25),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Center(
                    child: Text(
                      '확인',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.w700, color: Colors.white),
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
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
          '게시판 저장',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: Color(0xFF222222)),
        ),
        centerTitle: true,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFF0F0F0)),
        ),
      ),
      body: SingleChildScrollView(
        child: Column(
          children: [
            const SizedBox(height: 16),

            // 안내 카드
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 12,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Column(
                  children: const [
                    Text('📋', style: TextStyle(fontSize: 40)),
                    SizedBox(height: 8),
                    Text(
                      '분석 결과를 게시판에 저장해요',
                      style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800, color: Color(0xFF222222)),
                    ),
                    SizedBox(height: 4),
                    Text(
                      '제목과 비밀번호만 입력하면 완료!',
                      style: TextStyle(fontSize: 13, color: Color(0xFFAAAAAA)),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // 입력 폼 카드
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // 제목 입력
                    const Text(
                      '📝 제목',
                      style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
                    ),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _titleController,
                      style: const TextStyle(fontSize: 15, color: Color(0xFF333333)),
                      decoration: InputDecoration(
                        hintText: '예) 교차로 직진 vs 좌회전 사고',
                        hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFCCCCCC)),
                        filled: true,
                        fillColor: const Color(0xFFF8F8F8),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                      ),
                    ),

                    const SizedBox(height: 20),

                    // 비밀번호 입력
                    const Text(
                      '🔒 비밀번호',
                      style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      '게시글 조회 시 필요해요 (4자리 이상)',
                      style: TextStyle(fontSize: 12, color: Color(0xFFAAAAAA)),
                    ),
                    const SizedBox(height: 8),
                    TextField(
                      controller: _passwordController,
                      obscureText: _obscurePassword,
                      style: const TextStyle(fontSize: 15, color: Color(0xFF333333)),
                      decoration: InputDecoration(
                        hintText: '비밀번호 입력',
                        hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFCCCCCC)),
                        filled: true,
                        fillColor: const Color(0xFFF8F8F8),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        suffixIcon: GestureDetector(
                          onTap: () => setState(() => _obscurePassword = !_obscurePassword),
                          child: Icon(
                            _obscurePassword ? Icons.visibility_off : Icons.visibility,
                            size: 20,
                            color: const Color(0xFFBBBBBB),
                          ),
                        ),
                      ),
                    ),

                    const SizedBox(height: 12),

                    TextField(
                      controller: _passwordConfirmController,
                      obscureText: _obscureConfirm,
                      style: const TextStyle(fontSize: 15, color: Color(0xFF333333)),
                      decoration: InputDecoration(
                        hintText: '비밀번호 확인',
                        hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFCCCCCC)),
                        filled: true,
                        fillColor: const Color(0xFFF8F8F8),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        suffixIcon: GestureDetector(
                          onTap: () => setState(() => _obscureConfirm = !_obscureConfirm),
                          child: Icon(
                            _obscureConfirm ? Icons.visibility_off : Icons.visibility,
                            size: 20,
                            color: const Color(0xFFBBBBBB),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // 저장될 내용 미리보기
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '📄 저장될 내용 미리보기',
                      style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
                    ),
                    const SizedBox(height: 12),
                    _buildPreviewItem('🚗 사고 정황', _buildAccidentInfo()),
                    _buildPreviewItem('📊 과실 비율', _buildFaultRatio()),
                    _buildPreviewItem('📝 분석 결과', _buildAnalysisResult(), maxLines: 3),
                    _buildPreviewItem('📚 법적 근거', _buildLegalBasis(), maxLines: 2),
                    _buildPreviewItem('📋 사고 요약', _buildSummary(), maxLines: 3),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 20),

            // 저장 버튼
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: GestureDetector(
                onTap: _isLoading ? null : _saveToBoard,
                child: Container(
                  width: double.infinity,
                  height: 56,
                  decoration: BoxDecoration(
                    gradient: const LinearGradient(
                      colors: [Color(0xFF66BB6A), Color(0xFF43A047)],
                    ),
                    borderRadius: BorderRadius.circular(16),
                    boxShadow: [
                      BoxShadow(
                        color: const Color(0xFF43A047).withOpacity(0.25),
                        blurRadius: 12,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Center(
                    child: _isLoading
                        ? const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5),
                          )
                        : const Text(
                            '게시판에 저장하기 ✓',
                            style: TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: Colors.white),
                          ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _buildPreviewItem(String label, String content, {int maxLines = 2}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF888888)),
          ),
          const SizedBox(height: 4),
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFF8F8F8),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              content.isEmpty ? '정보 없음' : content,
              style: const TextStyle(fontSize: 13, color: Color(0xFF555555), height: 1.5),
              maxLines: maxLines,
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}