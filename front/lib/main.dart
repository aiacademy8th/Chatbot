import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import 'insurer.dart';
import 'photo.dart';
import 'chat.dart';
import 'emergency.dart'; 
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'dart:ui';
import 'package:animated_text_kit/animated_text_kit.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await dotenv.load(fileName: ".env"); // .env 파일 로드
  runApp(const AccidentHelperApp());
}

class AccidentHelperApp extends StatelessWidget {
  const AccidentHelperApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '사고 과실 도우미',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        fontFamily: 'NotoSans',
      ),
      home: const HomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

// StatefulWidget으로 변경 (앱 생명주기 감지)
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> with WidgetsBindingObserver {
  bool _justMadeCall = false;  // 방금 전화를 걸었는지 추적
  String _lastServiceName = '';  // 마지막으로 건 전화 (119 or 112)
  Key _typingTextKey = UniqueKey(); // 타이핑 애니메이션 재시작용 키

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

  // 앱 상태 변화 감지
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      // 화면 돌아올 때 애니메이션 재시작
      setState(() {
        _typingTextKey = UniqueKey();
      });

      // 전화 후 사진 안내
      if (_justMadeCall) {
        _justMadeCall = false;
        Future.delayed(const Duration(milliseconds: 500), () {
          if (mounted) _showPhotoPrompt(_lastServiceName);
        });
      }
    }
  }

  // 전화 걸기
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

  // 사진 촬영 안내 다이얼로그
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
            style: const TextStyle(
              fontSize: 16
              ),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 235, 243, 239), // 연두 느낌 배경
      appBar: AppBar(
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          '교통사고 AI 상담사',
          style: TextStyle(
            color: Color.fromARGB(255, 65, 77, 64),
            fontSize: 30,
            fontWeight: FontWeight.bold,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.article),
            tooltip: '게시판',
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('게시판 화면으로 이동 (준비중)')),
              );
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // 안내문 (위쪽)
          Align(
            alignment: Alignment.topLeft,
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              margin: const EdgeInsets.only(top:60, bottom: 20),
              decoration: BoxDecoration(
                // color: Colors.white.withOpacity(0.2),
                borderRadius: BorderRadius.circular(16),
                // border: Border.all(color: Colors.white.withOpacity(0.3)),
                // boxShadow: [
                //   BoxShadow(
                //     color: Colors.black.withOpacity(0.05),
                //     blurRadius: 10,
                //     offset: const Offset(0, 4),
                //   ),
                // ],
              ),
              child: DefaultTextStyle(
                style: const TextStyle(
                  fontSize: 14,
                  color: Color.fromARGB(221, 92, 90, 90),
                ),
                child: AnimatedTextKit(
                  key: _typingTextKey,
                  isRepeatingAnimation: false,
                  animatedTexts: [
                    TypewriterAnimatedText(
                      '걱정하지 마세요, 저희가 함께할게요.\n안전하고 편안한 사고 처리를 도와드려요.',
                      speed: const Duration(milliseconds: 100),
                      cursor: '|',
                    ),
                  ],
                ),
              ),
            ),
          ),

          // 버튼 3개 (화면 중앙 고정)
          Align(
            alignment: Alignment.center,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                GlassButton(
                  icon: Icons.warning_amber_rounded,
                  text: '긴급 전화',
                  accentColor: Colors.red,
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const EmergencyScreen(),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 20),
                GlassButton(
                  icon: Icons.phone_in_talk,
                  text: '보험사 연결',
                  accentColor: Color(0xFF2985FC),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const InsurerScreen(),
                      ),
                    );
                  },
                ),
                const SizedBox(height: 20),
                GlassButton(
                  icon: Icons.chat,
                  text: '사고 상담',
                  accentColor: Color.fromARGB(255, 107, 167, 18),
                  onPressed: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => const ChatbotScreen(
                          accidentPhotos: [],
                        ),
                      ),
                    );
                  },
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// 글래스모피즘 버튼
class GlassButton extends StatelessWidget {
  final VoidCallback onPressed;
  final IconData icon;
  final String text;
  final Color accentColor;

  const GlassButton({
    super.key,
    required this.onPressed,
    required this.icon,
    required this.text,
    required this.accentColor,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: MediaQuery.of(context).size.width - 40,
      height: 80,
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
          child: InkWell(
            onTap: onPressed,
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.25),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: Colors.white.withOpacity(0.4),
                ),
                boxShadow: [
                  BoxShadow(
                    color: accentColor.withOpacity(0.25),
                    blurRadius: 12,
                    offset: const Offset(0, 6),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(icon, size: 32, color: accentColor),
                  const SizedBox(width: 12),
                  Text(
                    text,
                    style: TextStyle(
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                      color: accentColor,
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
}
