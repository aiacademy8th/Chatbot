import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'photo.dart';

class InsurerScreen extends StatelessWidget {
  const InsurerScreen({super.key});

  // 보험사 목록 데이터
  final List<Map<String, String>> insurers = const [
    {'name': '삼성화재', 'phone': '1588-5114'},
    {'name': '현대해상', 'phone': '1588-5656'},
    {'name': 'DB손해보험', 'phone': '1588-0100'},
    {'name': '메리츠화재', 'phone': '1566-7711'},
    {'name': 'KB손해보험', 'phone': '1544-0114'},
    {'name': '한화손해보험', 'phone': '1566-8000'},
    {'name': 'AXA손해보험', 'phone': '1566-1566'},
    {'name': '롯데손해보험', 'phone': '1588-3344'},
    {'name': '흥국화재', 'phone': '1688-1688'},
    {'name': '캐롯자동차보험', 'phone': '1566-0300'},
    {'name': '하나손해보험', 'phone': '1566-3000'},
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
              title: Text('$insurerName 연결'),
              content: const Text('통화가 끝나셨나요?\n사고 차량 사진을 촬영해주세요.'),
              actions: [
                TextButton(
                  child: const Text('나중에'),
                  onPressed: () {
                    Navigator.pop(dialogContext);
                  },
                ),
                ElevatedButton(
                  child: const Text('사진 촬영하기'),
                  onPressed: () {
                    Navigator.pop(dialogContext);
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const PhotoScreen(),
                      ),
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
      // 연두색 배경
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [
              const Color.fromARGB(255, 255, 255, 255),
              const Color.fromARGB(255, 202, 216, 202),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
        ),
        padding: const EdgeInsets.all(16),
        child: SafeArea(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // 앱바 대체 영역 (뒤로가기 + 제목)
              Row(
                children: [
                  IconButton(
                    icon: const Icon(Icons.arrow_back, color: Colors.black87),
                    onPressed: () {
                      Navigator.pop(context);
                    },
                  ),
                  const SizedBox(width: 8),
                  const Text(
                    '보험사 연결',
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.bold,
                      color: Colors.black87,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // 안내 문구
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white.withOpacity(0.4)),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.white.withOpacity(0.25),
                      blurRadius: 8,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: const Row(
                  children: [
                    Icon(Icons.info_outline, color: Color.fromARGB(255, 0, 0, 0)),
                    SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        '보험사를 선택하시면 바로 전화 연결됩니다.',
                        style: TextStyle(
                          fontSize: 14,
                          color: Color.fromARGB(255, 0, 0, 0),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),

              // 보험사 리스트
              Expanded(
                child: ListView.builder(
                  itemCount: insurers.length,
                  itemBuilder: (context, index) {
                    final insurer = insurers[index];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 16),
                      child: GlassInsurerButton(
                        name: insurer['name']!,
                        phone: insurer['phone']!,
                        onPressed: () => _makePhoneCallAndNavigate(
                          context,
                          insurer['phone']!,
                          insurer['name']!,
                        ),
                      ),
                    );
                  },
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class GlassInsurerButton extends StatelessWidget {
  final String name;
  final String phone;
  final VoidCallback onPressed;

  const GlassInsurerButton({
    super.key,
    required this.name,
    required this.phone,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
        child: Container(
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.25),
            borderRadius: BorderRadius.circular(16),
            border: Border.all(color: Colors.white.withOpacity(0.4)),
            boxShadow: [
              BoxShadow(
                color: Colors.green.shade700.withOpacity(0.2),
                blurRadius: 12,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: ElevatedButton(
            onPressed: onPressed,
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.transparent,
              shadowColor: Colors.transparent,
              elevation: 0,
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              foregroundColor: Colors.black87,
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: Colors.green.shade50.withOpacity(0.6),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(
                    Icons.business,
                    color: Colors.green.shade700,
                    size: 28,
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    mainAxisAlignment: MainAxisAlignment.center,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        style: const TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.black87,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        phone,
                        style: TextStyle(
                          fontSize: 14,
                          color: Colors.grey.shade600,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  Icons.phone,
                  color: Colors.green.shade700,
                  size: 28,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
