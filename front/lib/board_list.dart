import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'board_view.dart';

class BoardListScreen extends StatefulWidget {
  const BoardListScreen({super.key});

  @override
  State<BoardListScreen> createState() => _BoardListScreenState();
}

class _BoardListScreenState extends State<BoardListScreen> {
  final String baseUrl = dotenv.env['BACKEND_URL'] ?? "";
  List<dynamic> _posts = [];
  bool _isLoading = true;
  String? _error;
  String _debugInfo = '';

  @override
  void initState() {
    super.initState();
    _fetchBoardList();
  }

  Future<void> _fetchBoardList() async {
    setState(() {
      _isLoading = true;
      _error = null;
    });

    debugPrint('📋 [게시판] baseUrl: $baseUrl');
    debugPrint('📋 [게시판] 요청 URL: $baseUrl/board/list');
    _debugInfo = 'URL: $baseUrl/board/list\n';

    try {
      final url = Uri.parse('$baseUrl/board/list');
      _debugInfo += '요청 시작...\n';
      final response = await http.get(url);

      _debugInfo += '상태코드: ${response.statusCode}\n';
      _debugInfo += '응답: ${utf8.decode(response.bodyBytes).substring(0, (utf8.decode(response.bodyBytes).length > 200 ? 200 : utf8.decode(response.bodyBytes).length))}\n';
      debugPrint('📋 [게시판] 상태코드: ${response.statusCode}');
      debugPrint('📋 [게시판] 응답 body: ${utf8.decode(response.bodyBytes)}');

      if (response.statusCode == 200) {
        final data = jsonDecode(utf8.decode(response.bodyBytes));
        debugPrint('📋 [게시판] 응답 타입: ${data.runtimeType}');

        List<dynamic> posts = [];
        if (data is List) {
          posts = data;
        } else if (data is Map<String, dynamic>) {
          posts = data['posts'] ?? data['data'] ?? data['list'] ?? data['board_list'] ?? data['items'] ?? [];
          if (posts.isEmpty && data.containsKey('result')) {
            final result = data['result'];
            posts = result is List ? result : [];
          }
        }

        debugPrint('📋 [게시판] 파싱된 게시글 수: ${posts.length}');
        setState(() {
          _posts = posts;
          _isLoading = false;
        });
      } else {
        debugPrint('📋 [게시판] 에러: ${response.statusCode}');
        setState(() {
          _error = '불러오기 실패: ${response.statusCode}';
          _isLoading = false;
        });
      }
    } catch (e, stackTrace) {
      debugPrint('📋 [게시판] 예외 발생: $e');
      debugPrint('📋 [게시판] 스택: $stackTrace');
      _debugInfo += '예외: $e\n';
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

  void _openPost(dynamic post) {
    final postId = post['post_id'] ?? post['id'] ?? 0;
    final title = post['accident_title'] ?? post['title'] ?? '제목 없음';

    _showPasswordDialog(postId, title);
  }

  void _showPasswordDialog(int postId, String title) {
    final passwordController = TextEditingController();
    bool obscure = true;

    showDialog(
      context: context,
      barrierColor: Colors.black.withOpacity(0.35),
      builder: (ctx) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return Dialog(
              backgroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              insetPadding: const EdgeInsets.symmetric(horizontal: 40),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text('🔒', style: TextStyle(fontSize: 36)),
                    const SizedBox(height: 8),
                    const Text(
                      '비밀번호 입력',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.w800, color: Color(0xFF222222)),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      title,
                      style: const TextStyle(fontSize: 13, color: Color(0xFFAAAAAA)),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 18),
                    TextField(
                      controller: passwordController,
                      obscureText: obscure,
                      style: const TextStyle(fontSize: 15, color: Color(0xFF333333)),
                      decoration: InputDecoration(
                        hintText: '비밀번호를 입력하세요',
                        hintStyle: const TextStyle(fontSize: 14, color: Color(0xFFCCCCCC)),
                        filled: true,
                        fillColor: const Color(0xFFF8F8F8),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: BorderSide.none,
                        ),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                        suffixIcon: GestureDetector(
                          onTap: () => setDialogState(() => obscure = !obscure),
                          child: Icon(
                            obscure ? Icons.visibility_off : Icons.visibility,
                            size: 20,
                            color: const Color(0xFFBBBBBB),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: GestureDetector(
                            onTap: () => Navigator.pop(ctx),
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 14),
                              decoration: BoxDecoration(
                                color: const Color(0xFFF5F5F5),
                                borderRadius: BorderRadius.circular(14),
                              ),
                              child: const Center(
                                child: Text(
                                  '취소',
                                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Color(0xFF999999)),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          flex: 2,
                          child: GestureDetector(
                            onTap: () {
                              if (passwordController.text.trim().isEmpty) return;
                              Navigator.pop(ctx);
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => BoardViewScreen(
                                    postId: postId,
                                    password: passwordController.text.trim(),
                                  ),
                                ),
                              );
                            },
                            child: Container(
                              padding: const EdgeInsets.symmetric(vertical: 14),
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
                              child: const Center(
                                child: Text(
                                  '확인 ✓',
                                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: Colors.white),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
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
          '게시판',
          style: TextStyle(fontSize: 20, fontWeight: FontWeight.w700, color: Color(0xFF222222)),
        ),
        centerTitle: true,
        actions: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: GestureDetector(
              onTap: _fetchBoardList,
              child: Container(
                width: 34,
                height: 34,
                decoration: const BoxDecoration(
                  color: Color(0xFFF0F0F0),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.refresh, size: 18, color: Color(0xFF555555)),
              ),
            ),
          ),
        ],
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(height: 1, color: const Color(0xFFF0F0F0)),
        ),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF43A047)))
          : _error != null
              ? _buildErrorView()
              : _posts.isEmpty
                  ? _buildEmptyView()
                  : _buildListView(),
    );
  }

  Widget _buildErrorView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('😵', style: TextStyle(fontSize: 48)),
          const SizedBox(height: 12),
          Text(
            _error!,
            style: const TextStyle(fontSize: 14, color: Color(0xFF999999)),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          GestureDetector(
            onTap: _fetchBoardList,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              decoration: BoxDecoration(
                color: const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Text(
                '다시 시도',
                style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF43A047)),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyView() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const Text('📭', style: TextStyle(fontSize: 56)),
          const SizedBox(height: 12),
          const Text(
            '아직 게시글이 없어요',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700, color: Color(0xFF555555)),
          ),
          const SizedBox(height: 4),
          const Text(
            '분석 결과를 저장하면 여기에 표시돼요',
            style: TextStyle(fontSize: 14, color: Color(0xFFAAAAAA)),
          ),
          const SizedBox(height: 20),
          // 디버그 정보 (나중에 제거)
          Container(
            margin: const EdgeInsets.symmetric(horizontal: 20),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F5F5),
              borderRadius: BorderRadius.circular(10),
              border: Border.all(color: const Color(0xFFE0E0E0)),
            ),
            child: Text(
              '🔍 디버그:\n$_debugInfo',
              style: const TextStyle(fontSize: 11, color: Color(0xFF999999), fontFamily: 'monospace'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildListView() {
    return RefreshIndicator(
      onRefresh: _fetchBoardList,
      color: const Color(0xFF43A047),
      child: ListView.builder(
        padding: const EdgeInsets.all(20),
        itemCount: _posts.length,
        itemBuilder: (context, index) {
          final post = _posts[index];
          return _buildPostCard(post, index);
        },
      ),
    );
  }

  Widget _buildPostCard(dynamic post, int index) {
    final postId = post['post_id'] ?? post['id'] ?? 0;
    final title = post['accident_title'] ?? post['title'] ?? '제목 없음';
    final date = _formatDate(post['created_at'] ?? post['date'] ?? '');

    // 게시글 번호별 이모지
    final emojis = ['📋', '📄', '📝', '📑', '🗂️'];
    final emoji = emojis[index % emojis.length];

    return GestureDetector(
      onTap: () => _openPost(post),
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: const Color(0xFFF0F0F0)),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.03),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: Row(
          children: [
            // 이모지 아이콘
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: const Color(0xFFF0F8F0),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Center(child: Text(emoji, style: const TextStyle(fontSize: 22))),
            ),
            const SizedBox(width: 14),

            // 제목 + 날짜
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 15,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF333333),
                    ),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    children: [
                      Text(
                        '#$postId',
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: Color(0xFF43A047),
                        ),
                      ),
                      if (date.isNotEmpty) ...[
                        const SizedBox(width: 8),
                        Text(
                          date,
                          style: const TextStyle(fontSize: 12, color: Color(0xFFBBBBBB)),
                        ),
                      ],
                    ],
                  ),
                ],
              ),
            ),

            // 화살표
            Container(
              width: 32,
              height: 32,
              decoration: const BoxDecoration(
                color: Color(0xFFF5F5F5),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_forward, size: 16, color: Color(0xFFAAAAAA)),
            ),
          ],
        ),
      ),
    );
  }
}