import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';
import 'dart:ui';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'result.dart';

class ChatbotScreen extends StatefulWidget {
  final List<File> accidentPhotos;

  const ChatbotScreen({
    super.key,
    required this.accidentPhotos,
  });

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  bool _chatStarted = false;
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _textController = TextEditingController();
  final String? openaiApiKey = dotenv.env['OPENAI_API_KEY'];

  final List<ChatMessage> _messages = [];
  int _currentStep = 0;
  final Map<String, dynamic> _userAnswers = {};
  bool _isLoading = false;

  // 색상 테마 정의 (Sky Blue Concept)
  static const Color glassWhite = Color(0x4DFFFFFF); // 30% 투명도 화이트
  static const Color glassBorder = Color(0x66FFFFFF); // 40% 투명도 테두리 (글래스 효과의 핵심)
  static const Color primaryBlue = Color(0xFF00ADEE);
  static const Color darkTextColor = Color(0xFF1A2A3A);

  final List<QuestionData> _questions = [
    QuestionData(
      id: 'time_blackbox',
      question: '사고 발생 시간과 블랙박스 유무를 선택해주세요.',
      type: QuestionType.timeAndBlackbox,
      options: ['있음', '없음'],
    ),
    QuestionData(
      id: 'location_type',
      question: '사고 장소는 어디인가요?',
      type: QuestionType.radio,
      options: ['교차로', '직선도로', '주차장', '기타/모름'],
    ),
    QuestionData(
      id: 'my_action',
      question: '사고 당시 나의 주행 상태는?',
      type: QuestionType.radio,
      options: ['직진', '좌회전', '우회전', '유턴', '정지', '후진', '주차', '기타'],
    ),
    QuestionData(
      id: 'opponent_action',
      question: '사고 당시 상대방의 주행 상태는?',
      type: QuestionType.radio,
      options: ['직진', '좌회전', '우회전', '유턴', '정지', '후진', '주차', '기타'],
    ),
    QuestionData(
      id: 'collision_part',
      question: '내 차량의 어느 부위가 충돌했나요?',
      type: QuestionType.multiSelect,
      options: ['앞면', '옆면(왼쪽)', '옆면(오른쪽)', '뒷면'],
    ),
    QuestionData(
      id: 'fault_factors',
      question: '과실 요소를 선택해주세요.',
      type: QuestionType.faultFactors,
      options: ['과속', '신호위반', '중앙선 침범', '안전거리 미확보', '끼어들기', '음주운전', '해당없음'],
    ),
    QuestionData(
      id: 'additional_info',
      question: '추가로 전달하고 싶은 정보가 있나요?',
      type: QuestionType.text,
      options: [],
    ),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _startChat();
    });
  }

  void _startChat() {
    if (_chatStarted) return;
    _chatStarted = true;
    _addBotMessage('안녕하세요! 🚗\n교통사고 과실 비율 분석을 도와드리겠습니다.');
    Future.delayed(const Duration(milliseconds: 600), () {
      if (_currentStep == 0 && _messages.length < 3) {
        _askCurrentQuestion();
      }
    });
  }

  @override
  void dispose() {
    _scrollController.dispose();
    _textController.dispose();
    super.dispose();
  }

  void _askCurrentQuestion() {
    if (_currentStep < _questions.length) {
      final question = _questions[_currentStep];
      _addBotMessage(question.question);
      if (question.type != QuestionType.text) {
        _addOptionsMessage(question);
      }
    } else {
      _analyzeWithGPT();
    }
  }

  void _addBotMessage(String text) {
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: false, timestamp: DateTime.now()));
    });
    _scrollToBottom();
  }

  void _addUserMessage(String text) {
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true, timestamp: DateTime.now()));
    });
    _scrollToBottom();
  }

  void _addOptionsMessage(QuestionData question) {
    setState(() {
      _messages.add(ChatMessage(
        text: '',
        isUser: false,
        timestamp: DateTime.now(),
        isOptions: true,
        questionData: question,
      ));
    });
    _scrollToBottom();
  }

  void _handleAnswer(String questionId, dynamic answer, String displayText) {
    _userAnswers[questionId] = answer;
    _addUserMessage(displayText);
    _currentStep++;
    Future.delayed(const Duration(milliseconds: 500), () {
      _askCurrentQuestion();
    });
  }

  Future<void> _analyzeWithGPT() async {
    _addBotMessage('입력하신 정보를 분석 중입니다... 🤔');
    if (!mounted) return;
    setState(() => _isLoading = true);

    try {
      final prompt = _buildAnalysisPrompt();
      final response = await _callGPT(prompt);

      if (mounted) {
        setState(() => _isLoading = false);
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultScreen(
              analysisResult: {
                'fault_ratio': _extractFaultRatio(response),
                'analysis': response,
                'references': _getRagReferences(),
              },
              userAnswers: _userAnswers,
            ),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _isLoading = false);
        _addBotMessage('분석 중 오류가 발생했습니다.\n오류: $e');
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.white.withOpacity(0.1),
        elevation: 0,
        flexibleSpace: ClipRect(
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
            child: Container(color: Colors.transparent),
          ),
        ),
        title: const Text(
          'AI 과실 분석 Chat',
          style: TextStyle(
            color: darkTextColor,
            fontWeight: FontWeight.bold,
          ),
        ),
        iconTheme: const IconThemeData(color: darkTextColor),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh, color: darkTextColor),
            onPressed: () {
              setState(() {
                _messages.clear();
                _userAnswers.clear();
                _currentStep = 0;
                _chatStarted = false;
              });
              _startChat();
            },
          ),
        ],
      ),
      body: Stack(
        children: [
          // 배경 이미지 또는 그라데이션
          Container(
            decoration: const BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [Color(0xFFE0F7FA), Color(0xFF80DEEA), Color(0xFF4DD0E1)],
              ),
            ),
          ),
          // 배경 위에 흐르는 듯한 원형 장식 (글래스모피즘 효과 극대화)
          Positioned(top: 100, left: -50, child: _buildCircle(200, Colors.white.withOpacity(0.3))),
          Positioned(bottom: 200, right: -80, child: _buildCircle(300, primaryBlue.withOpacity(0.2))),

          SafeArea(
            child: Column(
              children: [
                if (_currentStep < _questions.length)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: LinearProgressIndicator(
                        value: _currentStep / _questions.length,
                        minHeight: 8,
                        backgroundColor: Colors.white.withOpacity(0.3),
                        valueColor: const AlwaysStoppedAnimation<Color>(Colors.white),
                      ),
                    ),
                  ),
                Expanded(
                  child: ListView.builder(
                    controller: _scrollController,
                    padding: const EdgeInsets.all(16),
                    itemCount: _messages.length,
                    itemBuilder: (context, index) {
                      final message = _messages[index];
                      if (message.isOptions) return _buildOptionsWidget(message.questionData!);
                      return _buildMessageBubble(message);
                    },
                  ),
                ),
                if (_isLoading) const Padding(padding: EdgeInsets.all(16.0), child: CircularProgressIndicator(color: Colors.white)),
                if (_currentStep == 6) _buildTextInput(),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCircle(double size, Color color) {
    return Container(width: size, height: size, decoration: BoxDecoration(shape: BoxShape.circle, color: color));
  }

  Widget _buildMessageBubble(ChatMessage message) {
    bool isUser = message.isUser;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        decoration: BoxDecoration(
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 5,
              offset: const Offset(0, 2),
            )
          ],
        ),
        child: ClipRRect(
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(20),
            topRight: const Radius.circular(20),
            bottomLeft: Radius.circular(isUser ? 20 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 20),
          ),
          child: BackdropFilter(
            filter: ImageFilter.blur(sigmaX: 8, sigmaY: 8),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              constraints: BoxConstraints(maxWidth: MediaQuery.of(context).size.width * 0.75),
              decoration: BoxDecoration(
                color: isUser ? primaryBlue.withOpacity(0.8) : Colors.white.withOpacity(0.7),
                border: Border.all(color: Colors.white.withOpacity(0.3), width: 1),
              ),
              child: Text(
                message.text,
                style: TextStyle(
                  color: isUser ? Colors.white : darkTextColor,
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildOptionsWidget(QuestionData question) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 12),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(25),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 20, sigmaY: 20),
          child: Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: glassWhite,
              borderRadius: BorderRadius.circular(25),
              border: Border.all(color: glassBorder, width: 1.5),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (question.type == QuestionType.timeAndBlackbox) _buildTimeAndBlackboxOptions(question),
                if (question.type == QuestionType.radio) _buildRadioOptions(question),
                if (question.type == QuestionType.multiSelect) _buildMultiSelectOptions(question),
                if (question.type == QuestionType.faultFactors) _buildFaultFactorsOptions(question),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // Widget _glassCard({required Widget child}) {
  //   return Container(
  //     margin: const EdgeInsets.symmetric(vertical: 10),
  //     child: ClipRRect(
  //       borderRadius: BorderRadius.circular(25),
  //       child: BackdropFilter(
  //         filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
  //         child: Container(
  //           padding: const EdgeInsets.all(20),
  //           decoration: BoxDecoration(
  //             color: Colors.white.withOpacity(0.4),
  //             borderRadius: BorderRadius.circular(25),
  //             border: Border.all(color: Colors.white.withOpacity(0.5)),
  //           ),
  //           child: child,
  //         ),
  //       ),
  //     ),
  //   );
  // }

Widget _buildGlassButton({required String text, required VoidCallback onTap, bool isSelected = false, Widget? icon}) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(15),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        decoration: BoxDecoration(
          color: isSelected ? Colors.white.withOpacity(0.9) : Colors.white.withOpacity(0.3),
          borderRadius: BorderRadius.circular(15),
          border: Border.all(color: isSelected ? Colors.white : glassBorder, width: 1),
          boxShadow: isSelected ? [BoxShadow(color: Colors.black.withOpacity(0.1), blurRadius: 10)] : [],
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (icon != null) ...[icon, const SizedBox(width: 8)],
            Text(text, style: TextStyle(color: darkTextColor, fontWeight: isSelected ? FontWeight.w800 : FontWeight.w600)),
          ],
        ),
      ),
    );
  }

  Widget _buildRadioOptions(QuestionData question) {
      return Wrap(
        spacing: 10,
        runSpacing: 10,
        children: question.options.map((option) => _buildGlassButton(text: option, onTap: () => _handleAnswer(question.id, option, option))).toList(),
      );
  }

  Widget _buildMultiSelectOptions(QuestionData question) {
    final Set<String> selected = {};
    return StatefulBuilder(builder: (context, setStateLocal) {
      return Column(
        children: [
          Wrap(
            spacing: 10,
            runSpacing: 10,
            children: question.options.map((opt) {
              return _buildGlassButton(
                text: opt,
                isSelected: selected.contains(opt),
                onTap: () => setStateLocal(() => selected.contains(opt) ? selected.remove(opt) : selected.add(opt)),
              );
            }).toList(),
          ),
          const SizedBox(height: 15),
          _buildGlassButton(
            text: '선택 완료',
            isSelected: true,
            onTap: selected.isEmpty ? () {} : () => _handleAnswer(question.id, selected.toList(), selected.join(', ')),
          ),
        ],
      );
    });
  }

  Widget _buildFaultFactorsOptions(QuestionData question) {
    final Set<String> my = {};
    final Set<String> op = {};
    return StatefulBuilder(builder: (context, setStateLocal) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('나의 과실', style: TextStyle(fontWeight: FontWeight.bold, color: darkTextColor)),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 8, children: question.options.map((opt) => _buildGlassButton(text: opt, isSelected: my.contains(opt), onTap: () => setStateLocal(() => my.contains(opt) ? my.remove(opt) : my.add(opt)))).toList()),
          const SizedBox(height: 16),
          const Text('상대의 과실', style: TextStyle(fontWeight: FontWeight.bold, color: darkTextColor)),
          const SizedBox(height: 8),
          Wrap(spacing: 8, runSpacing: 8, children: question.options.map((opt) => _buildGlassButton(text: opt, isSelected: op.contains(opt), onTap: () => setStateLocal(() => op.contains(opt) ? op.remove(opt) : op.add(opt)))).toList()),
          const SizedBox(height: 20),
          Center(child: _buildGlassButton(text: '분석 시작', isSelected: true, onTap: () => _handleAnswer(question.id, {'my': my.toList(), 'opponent': op.toList()}, '입력 완료'))),
        ],
      );
    });
  }

  Widget _buildTextInput() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(30),
        child: BackdropFilter(
          filter: ImageFilter.blur(sigmaX: 15, sigmaY: 15),
          child: Container(
            color: Colors.white.withOpacity(0.2),
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    style: const TextStyle(color: darkTextColor),
                    decoration: const InputDecoration(hintText: '메시지를 입력하세요...', border: InputBorder.none, contentPadding: EdgeInsets.symmetric(horizontal: 20)),
                  ),
                ),
                IconButton(icon: const Icon(Icons.send, color: darkTextColor), onPressed: () {
                  final text = _textController.text.trim();
                  _handleAnswer(_questions[_currentStep].id, text.isEmpty ? '없음' : text, text.isEmpty ? '없음' : text);
                  _textController.clear();
                }),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildTimeAndBlackboxOptions(QuestionData question) {
    return _buildGlassButton(
      text: '시간 및 블랙박스 설정',
      icon: const Icon(Icons.watch_later_outlined, color: darkTextColor, size: 20),
      onTap: () async {
        final TimeOfDay? picked = await showTimePicker(context: context, initialTime: TimeOfDay.now());
        if (picked != null) {
          final timeStr = '${picked.hour}:${picked.minute.toString().padLeft(2, '0')}';
          _showBlackboxDialog(question, timeStr);
        }
      },
    );
  }

  void _showBlackboxDialog(QuestionData question, String timeStr) {
    showDialog(
      context: context,
      builder: (context) => BackdropFilter(
        filter: ImageFilter.blur(sigmaX: 10, sigmaY: 10),
        child: AlertDialog(
          backgroundColor: Colors.white.withOpacity(0.8),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
          title: const Text('블랙박스 유무'),
          actions: ['있음', '없음'].map((opt) => TextButton(onPressed: () { Navigator.pop(context); _handleAnswer(question.id, {'time': timeStr, 'blackbox': opt}, '시간: $timeStr, 블박: $opt'); }, child: Text(opt))).toList(),
        ),
      ),
    );
  }

  String _buildAnalysisPrompt() {
    final buffer = StringBuffer('교통사고 과실 분석 전문가로서 다음 데이터를 분석해주세요:\n');
    _userAnswers.forEach((key, value) => buffer.writeln('$key: $value'));
    buffer.writeln('\n반드시 다음 형식을 지켜 한국어로 답변하세요:\n1. 예상 과실 비율 (예: 70:30)\n2. 상세 판단 근거\n3. 향후 대응 및 주의사항');
    return buffer.toString();
  }

  Future<String> _callGPT(String prompt) async {
    final response = await http.post(
      Uri.parse('https://api.openai.com/v1/chat/completions'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $openaiApiKey'
      },
      body: jsonEncode({
        'model': 'gpt-4o-mini',
        'messages': [
          {'role': 'system', 'content': '당신은 대한민국 교통사고 손해사정사 및 과실 비율 전문가입니다.'},
          {'role': 'user', 'content': prompt}
        ],
      }),
    );
    if (response.statusCode == 200) {
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      return data['choices'][0]['message']['content'];
    }
    throw Exception('분석 서버 응답 실패');
  }

  String _extractFaultRatio(String res) {
    final match = RegExp(r'(\d+)\s*:\s*(\d+)').firstMatch(res);
    return match != null ? '${match.group(1)}:${match.group(2)}' : '과실 판단 중';
  }

  List<Map<String, String>> _getRagReferences() => [
    {'source': '도로교통법 제13조', 'content': '차마의 통행 방법 관련'},
    {'source': '과실비율 인정기준', 'content': '손해보험협회 표준 데이터 참고'}
  ];

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 300), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }
}
class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  final bool isOptions;
  final QuestionData? questionData;
  
  ChatMessage({
    required this.text, 
    required this.isUser, 
    required this.timestamp, 
    this.isOptions = false, 
    this.questionData
  });
}

class QuestionData {
  final String id;
  final String question;
  final QuestionType type;
  final List<String> options;
  
  QuestionData({
    required this.id, 
    required this.question, 
    required this.type, 
    required this.options
  });
}

enum QuestionType { 
  radio, 
  multiSelect, 
  timeAndBlackbox, 
  faultFactors, 
  text 
}