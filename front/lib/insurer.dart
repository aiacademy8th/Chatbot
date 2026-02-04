import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'photo.dart';  // ⭐ 추가: photo.dart import

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

  // ⭐ 수정: 전화 걸기 후 사진 촬영 화면으로 이동
Future<void> _makePhoneCallAndNavigate(
  BuildContext context,
  String phoneNumber,
  String insurerName,  // ⭐ 보험사 이름 추가
) async {
  final Uri launchUri = Uri(scheme: 'tel', path: phoneNumber);
  
  if (await canLaunchUrl(launchUri)) {
    await launchUrl(launchUri);
    
    // ⭐ 전화 후 다이얼로그 표시
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
                  // ⭐ PhotoScreen으로 이동!
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
      appBar: AppBar(
        title: const Text(
          '보험사 연결',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.bold,
          ),
        ),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () {
            Navigator.pop(context);
          },
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // 안내 문구
            Container(
              padding: const EdgeInsets.all(16.0),
              decoration: BoxDecoration(
                color: Colors.blue.shade50,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.blue.shade200),
              ),
              child: const Row(
                children: [
                  Icon(Icons.info_outline, color: Colors.blue),
                  SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      '보험사를 선택하시면 바로 전화 연결됩니다.',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.black87,
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
                    padding: const EdgeInsets.only(bottom: 12.0),
                    child: InsurerButton(
                      name: insurer['name']!,
                      phone: insurer['phone']!,
                      onPressed: () => _makePhoneCallAndNavigate(
                        context,
                        insurer['phone']!,
                        insurer['name']!,
                      ),  // ⭐ 수정: 보험사 이름도 전달
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// 보험사 버튼 위젯
class InsurerButton extends StatelessWidget {
  final String name;
  final String phone;
  final VoidCallback onPressed;

  const InsurerButton({
    super.key,
    required this.name,
    required this.phone,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      // height: 80,
      constraints: const BoxConstraints(minHeight: 80),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 1,
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.white,
          foregroundColor: Colors.black87,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
            side: BorderSide(color: Colors.grey.shade300),
          ),
          elevation: 0,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
        ),
        child: Row(
          children: [
            // 보험사 아이콘
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: Colors.green.shade50,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Icon(
                Icons.business,
                color: Colors.green.shade700,
                size: 28,
              ),
            ),
            const SizedBox(width: 16),
            
            // 보험사 이름과 전화번호
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
            
            // 전화 아이콘
            Icon(
              Icons.phone,
              color: Colors.green.shade700,
              size: 28,
            ),
          ],
        ),
      ),
    );
  }
}