import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'photo.dart';

class InsurerScreen extends StatelessWidget {
  const InsurerScreen({super.key});

  // 보험사 목록 데이터 (이니셜 + 컬러 포함)
  static final List<Map<String, dynamic>> insurers = [
    {'name': '삼성화재', 'phone': '1588-5114', 'initial': '삼', 'colors': [const Color(0xFF66BB6A), const Color(0xFF43A047)]},
    {'name': '현대해상', 'phone': '1588-5656', 'initial': '현', 'colors': [const Color(0xFF42A5F5), const Color(0xFF1E88E5)]},
    {'name': 'DB손해보험', 'phone': '1588-0100', 'initial': 'D', 'colors': [const Color(0xFFFFA726), const Color(0xFFFB8C00)]},
    {'name': '메리츠화재', 'phone': '1566-7711', 'initial': '메', 'colors': [const Color(0xFFAB47BC), const Color(0xFF8E24AA)]},
    {'name': 'KB손해보험', 'phone': '1544-0114', 'initial': 'K', 'colors': [const Color(0xFF66BB6A), const Color(0xFF43A047)]},
    {'name': '한화손해보험', 'phone': '1566-8000', 'initial': '한', 'colors': [const Color(0xFFEF5350), const Color(0xFFE53935)]},
    {'name': 'AXA손해보험', 'phone': '1566-1566', 'initial': 'A', 'colors': [const Color(0xFF26C6DA), const Color(0xFF00ACC1)]},
    {'name': '롯데손해보험', 'phone': '1588-3344', 'initial': '롯', 'colors': [const Color(0xFFFFA726), const Color(0xFFFB8C00)]},
    {'name': '흥국화재', 'phone': '1688-1688', 'initial': '흥', 'colors': [const Color(0xFF42A5F5), const Color(0xFF1E88E5)]},
    {'name': '캐롯자동차보험', 'phone': '1566-0300', 'initial': '캐', 'colors': [const Color(0xFFEF5350), const Color(0xFFE53935)]},
    {'name': '하나손해보험', 'phone': '1566-3000', 'initial': '하', 'colors': [const Color(0xFFAB47BC), const Color(0xFF8E24AA)]},
  ];

  Future<void> _makePhoneCallAndNavigate(
    BuildContext context,
    String phoneNumber,
    String insurerName,
  ) async {
    final Uri launchUri = Uri(scheme: 'tel', path: phoneNumber);

    if (await canLaunchUrl(launchUri)) {
      await launchUrl(launchUri);

      if (context.mounted) {
        showDialog(
          context: context,
          builder: (BuildContext dialogContext) {
            return AlertDialog(
              backgroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
              title: Text('$insurerName 연결'),
              content: const Text(
                '통화가 끝나셨나요?\n사고 차량 사진을 촬영해주세요.',
                style: TextStyle(fontSize: 16),
              ),
              actions: [
                TextButton(
                  child: const Text('나중에'),
                  onPressed: () => Navigator.pop(dialogContext),
                ),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF43A047),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                  ),
                  child: const Text('사진 촬영하기'),
                  onPressed: () {
                    Navigator.pop(dialogContext);
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => const PhotoScreen()),
                    );
                  },
                ),
              ],
            );
          },
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF8F8F6),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        foregroundColor: const Color(0xFF222222),
        leading: Padding(
          padding: const EdgeInsets.all(8),
          child: GestureDetector(
            onTap: () => Navigator.pop(context),
            child: Container(
              decoration: BoxDecoration(
                color: const Color(0xFFF0F0F0),
                shape: BoxShape.circle,
              ),
              child: const Icon(Icons.arrow_back, size: 18, color: Color(0xFF555555)),
            ),
          ),
        ),
        title: const Text(
          '보험사 연결',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Color(0xFF222222),
          ),
        ),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 16),

            // 상단 안내 카드
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                ),
                child: Row(
                  children: const [
                    Text('📞', style: TextStyle(fontSize: 22)),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          children: [
                            TextSpan(
                              text: '탭하면 ',
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF555555),
                              ),
                            ),
                            TextSpan(
                              text: '바로 전화 연결됩니다',
                              style: TextStyle(
                                fontSize: 15,
                                color: Color(0xFF888888),
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

            const SizedBox(height: 20),

            // 섹션 라벨
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 22),
              child: Text(
                '보험사 목록',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: Color(0xFFAAAAAA),
                ),
              ),
            ),

            const SizedBox(height: 10),

            // 보험사 통합 카드 리스트
            Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: Container(
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
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: ListView.separated(
                      padding: EdgeInsets.zero,
                      itemCount: insurers.length,
                      separatorBuilder: (context, index) => const Divider(
                        height: 1,
                        thickness: 1,
                        color: Color(0xFFF5F5F5),
                        indent: 68,
                      ),
                      itemBuilder: (context, index) {
                        final insurer = insurers[index];
                        return _buildInsurerItem(
                          context: context,
                          name: insurer['name'] as String,
                          phone: insurer['phone'] as String,
                          initial: insurer['initial'] as String,
                          colors: insurer['colors'] as List<Color>,
                        );
                      },
                    ),
                  ),
                ),
              ),
            ),

            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildInsurerItem({
    required BuildContext context,
    required String name,
    required String phone,
    required String initial,
    required List<Color> colors,
  }) {
    return GestureDetector(
      onTap: () => _makePhoneCallAndNavigate(context, phone, name),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        color: Colors.white,
        child: Row(
          children: [
            // 이니셜 아이콘
            Container(
              width: 40,
              height: 40,
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: colors,
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(11),
              ),
              child: Center(
                child: Text(
                  initial,
                  style: const TextStyle(
                    fontSize: 17,
                    fontWeight: FontWeight.w800,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
            const SizedBox(width: 14),
            // 텍스트
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: const TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF333333),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    phone,
                    style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFFBBBBBB),
                    ),
                  ),
                ],
              ),
            ),
            // 전화 필 버튼
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFF5F5F5),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: const [
                  Icon(Icons.call, size: 14, color: Color(0xFF43A047)),
                  SizedBox(width: 4),
                  Text(
                    '전화',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: Color(0xFF43A047),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}