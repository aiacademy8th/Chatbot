import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'dart:ui';
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
          title: Row(
            children: [
              Icon(Icons.camera_alt, color: Colors.orange.shade700),
              const SizedBox(width: 8),
              const Text('사진 촬영'),
            ],
          ),
          content: Text(
            '$serviceName 통화가 끝나셨나요?\n사고 현장 사진을 촬영하시겠습니까?',
            style: const TextStyle(fontSize: 16),
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

  Widget _glassButton({required String label, required IconData icon, required VoidCallback onTap, required Color color}) {
    return SizedBox(
      width: double.infinity,
      height: 80,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
          child: InkWell(
            onTap: onTap,
            splashColor: Colors.white.withOpacity(0.2),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.25),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.4)),
                boxShadow: [
                  BoxShadow(
                    color: color.withOpacity(0.35),
                    blurRadius: 12,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(icon, size: 32, color: color),
                  const SizedBox(width: 12),
                  Text(
                    label,
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: color,
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('긴급 전화'),
        backgroundColor: Colors.transparent,
        elevation: 0,
        foregroundColor: Colors.black,
      ),
      body: Container(
        color: Colors.white,  // 배경 흰색
        width: double.infinity,
        height: double.infinity,
        child: Stack(
          alignment: Alignment.center,
          children: [
            // 가운데 초록 원형 그라데이션
            Container(
              width: 25000,
              height: 25000,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                gradient: RadialGradient(
                  colors: [
                    Colors.green.shade300.withOpacity(0.4),
                    Colors.green.shade100.withOpacity(0.0),
                  ],
                  center: Alignment.center,
                  radius: 0.7,
                ),
              ),
            ),
            // 버튼 영역
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  _glassButton(
                    label: '119',
                    icon: Icons.local_hospital,
                    color: Colors.red.shade700,
                    onTap: () => _makePhoneCall('119', '119'),
                  ),
                  const SizedBox(height: 40),
                  _glassButton(
                    label: '112',
                    icon: Icons.local_police,
                    color: Colors.red.shade700,
                    onTap: () => _makePhoneCall('112', '112'),
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
