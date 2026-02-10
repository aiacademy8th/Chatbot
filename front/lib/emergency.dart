import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'photo.dart';

class EmergencyScreen extends StatefulWidget {
  const EmergencyScreen({super.key});

  @override
  State<EmergencyScreen> createState() => _EmergencyScreenState();
}

class _EmergencyScreenState extends State<EmergencyScreen> with WidgetsBindingObserver {
  bool _justMadeCall = false;
  String _lastServiceName = '';

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    super.didChangeAppLifecycleState(state);

    if (state == AppLifecycleState.resumed && _justMadeCall) {
      _justMadeCall = false;

      Future.delayed(const Duration(milliseconds: 500), () {
        if (mounted) {
          _showPhotoPrompt(_lastServiceName);
        }
      });
    }
  }

  Future<void> _makePhoneCall(String phoneNumber, String serviceName) async {
    final Uri launchUri = Uri(scheme: 'tel', path: phoneNumber);
    try {
      if (await canLaunchUrl(launchUri)) {
        setState(() {
          _justMadeCall = true;
          _lastServiceName = serviceName;
        });
        await launchUrl(launchUri);
      } else {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('전화를 걸 수 없습니다: $phoneNumber')),
          );
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('오류 발생: $e')),
        );
      }
    }
  }

  void _showPhotoPrompt(String serviceName) {
    showDialog(
      context: context,
      barrierDismissible: true,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          backgroundColor: Colors.white,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: Row(
            children: [
              Icon(Icons.camera_alt, color: Colors.orange.shade700),
              const SizedBox(width: 8),
              const Text('사진 촬영'),
            ],
          ),
          content: Text(
            '$serviceName 통화가 끝나셨나요?\n사고 현장 사진을 촬영하시겠습니까?',
            style: const TextStyle(fontSize: 18),
          ),
          actions: [
            TextButton(
              child: const Text('나중에'),
              onPressed: () => Navigator.pop(dialogContext),
            ),
            ElevatedButton.icon(
              icon: const Icon(Icons.camera_alt, size: 20),
              label: const Text('사진 촬영하기'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.orange,
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFAFAF8),
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
                color: const Color(0xFFF5F5F5),
                borderRadius: BorderRadius.circular(10),
              ),
              child: const Icon(Icons.arrow_back, size: 18, color: Color(0xFF555555)),
            ),
          ),
        ),
        title: const Text(
          '긴급 전화',
          style: TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w700,
            color: Color(0xFF222222),
          ),
        ),
        centerTitle: false,
      ),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              // 히어로 카드
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 24),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(22),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withOpacity(0.05),
                      blurRadius: 12,
                      offset: const Offset(0, 2),
                    ),
                  ],
                ),
                child: Stack(
                  children: [
                    // 장식 blob
                    Positioned(
                      top: 0,
                      right: 0,
                      child: Container(
                        width: 30,
                        height: 30,
                        decoration: const BoxDecoration(
                          color: Color(0xFFFFEBEE),
                          shape: BoxShape.circle,
                        ),
                      ),
                    ),
                    Positioned(
                      bottom: 0,
                      left: 0,
                      child: Container(
                        width: 18,
                        height: 18,
                        decoration: const BoxDecoration(
                          color: Color(0xFFFFF3E0),
                          shape: BoxShape.circle,
                        ),
                      ),
                    ),
                    Positioned(
                      top: 26,
                      left: 0,
                      child: Transform.rotate(
                        angle: 0.35,
                        child: Container(
                          width: 12,
                          height: 12,
                          decoration: BoxDecoration(
                            color: const Color(0xFFE3F2FD),
                            borderRadius: BorderRadius.circular(4),
                          ),
                        ),
                      ),
                    ),
                    // 메인 내용
                    Center(
                      child: Column(
                        children: const [
                          Text('🆘', style: TextStyle(fontSize: 42)),
                          SizedBox(height: 12),
                          Text(
                            '긴급 상황인가요?',
                            style: TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w800,
                              color: Color(0xFF222222),
                            ),
                          ),
                          SizedBox(height: 6),
                          Text(
                            '아래 버튼을 눌러 바로 연결하세요',
                            style: TextStyle(
                              fontSize: 15,
                              color: Color(0xFFAAAAAA),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              const Spacer(),

              // 119 버튼
              _buildEmergencyButton(
                emoji: '🚒',
                number: '119',
                subtitle: '소방 · 구급 · 구조',
                emojiBgColor: const Color(0xFFFFEBEE),
                arrowBgColor: const Color(0xFFFFEBEE),
                arrowIconColor: const Color(0xFFE53935),
                onTap: () => _makePhoneCall('119', '119'),
              ),

              const SizedBox(height: 14),

              // 112 버튼
              _buildEmergencyButton(
                emoji: '🚔',
                number: '112',
                subtitle: '경찰 신고',
                emojiBgColor: const Color(0xFFE3F2FD),
                arrowBgColor: const Color(0xFFE3F2FD),
                arrowIconColor: const Color(0xFF1565C0),
                onTap: () => _makePhoneCall('112', '112'),
              ),

              const Spacer(),

              // 하단 팁
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFFF0F0F0)),
                ),
                child: Row(
                  children: const [
                    Text('💡', style: TextStyle(fontSize: 20)),
                    SizedBox(width: 10),
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          children: [
                            TextSpan(
                              text: '팁! ',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF666666),
                              ),
                            ),
                            TextSpan(
                              text: '부상자가 있다면 119를 먼저 연락하세요',
                              style: TextStyle(
                                fontSize: 14,
                                color: Color(0xFF999999),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmergencyButton({
    required String emoji,
    required String number,
    required String subtitle,
    required Color emojiBgColor,
    required Color arrowBgColor,
    required Color arrowIconColor,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: double.infinity,
        height: 84,
        padding: const EdgeInsets.symmetric(horizontal: 20),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
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
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                color: emojiBgColor,
                borderRadius: BorderRadius.circular(16),
              ),
              child: Center(
                child: Text(emoji, style: const TextStyle(fontSize: 26)),
              ),
            ),
            const SizedBox(width: 16),
            // 텍스트
            Expanded(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    number,
                    style: const TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.w800,
                      color: Color(0xFF333333),
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    subtitle,
                    style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFFBBBBBB),
                    ),
                  ),
                ],
              ),
            ),
            // 전화 아이콘
            Container(
              width: 36,
              height: 36,
              decoration: BoxDecoration(
                color: arrowBgColor,
                shape: BoxShape.circle,
              ),
              child: Icon(
                Icons.call,
                size: 18,
                color: arrowIconColor,
              ),
            ),
          ],
        ),
      ),
    );
  }
}