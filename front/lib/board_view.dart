import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class BoardViewScreen extends StatefulWidget {
  final int postId;
  final String password;

  const BoardViewScreen({
    super.key,
    required this.postId,
    required this.password,
  });

  @override
  State<BoardViewScreen> createState() => _BoardViewScreenState();
}

class _BoardViewScreenState extends State<BoardViewScreen> {
  final String baseUrl = dotenv.env['BACKEND_URL'] ?? "";
  Map<String, dynamic>? _postData;
  bool _isLoading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _fetchPostDetail();
  }

  Future<void> _fetchPostDetail() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    try {
      final url = Uri.parse('$baseUrl/board/view');
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'post_id': widget.postId,
          'password': widget.password,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        setState(() {
          _postData = data is Map<String, dynamic> ? data : {};
          _isLoading = false;
        });
      } else {
        final errorBody = utf8.decode(response.bodyBytes);
        setState(() {
          _error = response.statusCode == 403 || response.statusCode == 401
              ? '비밀번호가 일치하지 않습니다 🔒'
              : '불러오기 실패: ${response.statusCode}';
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        _error = '네트워크 오류: $e';
        _isLoading = false;
      });
    }
  }

  String _formatDate(String? dateStr) {
    if (dateStr == null || dateStr.isEmpty) return '';
    try {
      final dt = DateTime.parse(dateStr);
      return '${dt.year}.${dt.month.toString().padLeft(2, '0')}.${dt.day.toString().padLeft(2, '0')} ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}';
    } catch (_) {
      return dateStr;
    }
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
        title: Text(
          '게시글 #${widget.postId}',
          style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: Color(0xFF222222)),
        ),
        centerTitle: true,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFF0F0F0)),
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF43A047)))
          : _error != null
              ? _buildErrorView()
              : _buildDetailView(),
    );
  }

  Widget _buildErrorView() {
    final isPasswordError = _error?.contains('비밀번호') ?? false;
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(isPasswordError ? '🔒' : '😵', style: const TextStyle(fontSize: 48)),
          const SizedBox(height: 12),
          Text(
            _error!,
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600, color: Color(0xFF555555)),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              decoration: BoxDecoration(
                color: const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                '돌아가기',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Color(0xFF43A047)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDetailView() {
    final data = _postData!;
    final title = data['accident_title'] ?? data['title'] ?? '제목 없음';
    final date = _formatDate(data['created_at'] ?? data['date'] ?? '');
    final accidentInfo = data['accident_info'] ?? '';
    final faultRatio = data['fault_ratio'] ?? '';
    final analysisResult = data['analysis_result'] ?? '';
    final legalBasis = data['legal_basis'] ?? '';
    final accidentSummary = data['accident_summary'] ?? '';

    return SingleChildScrollView(
      child: Column(
        children: [
          const SizedBox(height: 16),

          // 제목 헤더 카드
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
                  const Text('📋', style: TextStyle(fontSize: 36)),
                  const SizedBox(height: 8),
                  Text(
                    title,
                    style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF222222)),
                    textAlign: TextAlign.center,
                  ),
                  if (date.isNotEmpty) ...[
                    const SizedBox(height: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: const Color(0xFFF5F5F5),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        '🕐 $date',
                        style: const TextStyle(fontSize: 12, color: Color(0xFFAAAAAA)),
                      ),
                    ),
                  ],
                  const SizedBox(height: 6),
                  Text(
                    '#${widget.postId}',
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: Color(0xFF43A047)),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 14),

          // 과실 비율
          if (faultRatio.isNotEmpty)
            _buildDetailCard('📊', '과실 비율', faultRatio, highlightColor: const Color(0xFFE3F2FD)),

          // 사고 정황
          if (accidentInfo.isNotEmpty)
            _buildDetailCard('🚗', '사고 정황', accidentInfo),

          // 분석 결과
          if (analysisResult.isNotEmpty)
            _buildDetailCard('📝', '분석 결과', analysisResult),

          // 법적 근거
          if (legalBasis.isNotEmpty)
            _buildDetailCard('📚', '법적 근거', legalBasis, highlightColor: const Color(0xFFFFFDE7)),

          // 사고 요약
          if (accidentSummary.isNotEmpty)
            _buildDetailCard('📋', '사고 요약', accidentSummary, highlightColor: const Color(0xFFE8F5E9)),

          const SizedBox(height: 14),

          // 주의사항
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
                      '본 분석 결과는 AI 기반 예측이며, 실제 판단과 다를 수 있습니다.',
                      style: TextStyle(fontSize: 12, color: Color(0xFFAAAAAA), height: 1.5),
                    ),
                  ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _buildDetailCard(String emoji, String title, String content, {Color? highlightColor}) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 0, 20, 14),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(18),
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
            Row(
              children: [
                Text(emoji, style: const TextStyle(fontSize: 18)),
                const SizedBox(width: 8),
                Text(
                  title,
                  style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
                ),
              ],
            ),
            const SizedBox(height: 12),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: highlightColor ?? const Color(0xFFF8F8F8),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                content,
                style: const TextStyle(fontSize: 14, color: Color(0xFF555555), height: 1.7),
              ),
            ),
          ],
        ),
      ),
    );
  }
}