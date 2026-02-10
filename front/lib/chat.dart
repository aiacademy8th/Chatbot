import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'result.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class ChatbotScreen extends StatefulWidget {
  final List<XFile> accidentPhotos;
  final String? threadId;
  final bool initialChatMode;

  const ChatbotScreen({
    super.key,
    required this.accidentPhotos,
    this.threadId,
    this.initialChatMode = false,
  });

  @override
  State<ChatbotScreen> createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  bool _chatStarted = false;
  final ScrollController _scrollController = ScrollController();
  final TextEditingController _textController = TextEditingController();

  final openaiApiKey = '';

  final List<ChatMessage> _messages = [];
  int _currentStep = 0;
  final Map<String, dynamic> _userAnswers = {};
  bool _isLoading = false;
  bool _inChatMode = false;
  String? _currentThreadId;

  // final String baseUrl = dotenv.env['BACKEND_URL'] ?? "";

  String get _baseUrl {
    final url = dotenv.env['BACKEND_URL'] ?? "";
    if (url.isEmpty) {
      debugPrint("⚠️ 경고: BACKEND_URL이 설정되지 않았습니다.");
      // 개발 단계에서 문제를 빨리 파악할 수 있도록 에러를 던지거나 기본값 설정
    }
    return url;
  }

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
      question: '내 차량의 어느 부위가 충돌했나요? (중복 선택 가능)',
      type: QuestionType.multiSelect,
      options: ['앞면', '옆면(왼쪽)', '옆면(오른쪽)', '뒷면'],
    ),
    QuestionData(
      id: 'fault_factors',
      question: '과실 요소를 선택해주세요. (중복 선택 가능)',
      type: QuestionType.faultFactors,
      options: ['과속', '신호위반', '중앙선 침범', '안전거리 미확보', '끼어들기', '음주운전', '해당없음'],
    ),
    QuestionData(
      id: 'additional_info',
      question: '추가로 전달하고 싶은 정보가 있나요? (선택사항)',
      type: QuestionType.text,
      options: [],
    ),
  ];

  @override
  void initState() {
    super.initState();

    if (widget.initialChatMode && widget.threadId != null) {
      _currentThreadId = widget.threadId;
      _inChatMode = true;
      _chatStarted = true;
      _addBotMessage('분석 결과를 바탕으로 추가 질문을 해주세요! 🤔');
    } else {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_messages.isEmpty) {
          _startChat();
        }
      });
    }
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
      _messages.add(ChatMessage(
        text: text,
        isUser: false,
        timestamp: DateTime.now(),
      ));
    });
    _scrollToBottom();
  }

  void _addUserMessage(String text) {
    setState(() {
      _messages.add(ChatMessage(
        text: text,
        isUser: true,
        timestamp: DateTime.now(),
      ));
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

    setState(() {
      _isLoading = true;
    });

    try {
      final prompt = _buildAnalysisPrompt();
      final backendResponse = await _callBackendAnalysis(prompt);

      final analysisResultData = backendResponse['result'] as Map<String, dynamic>;
      final threadId = backendResponse['thread_id'] as String? ?? '';

      if (mounted) {
        setState(() {
          _isLoading = false;
        });

        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => ResultScreen(
              analysisResult: backendResponse,
              userAnswers: _userAnswers,
              threadId: threadId,
            ),
          ),
        );
      }
    } catch (e, stacktrace) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        
        _addBotMessage('죄송합니다. 분석 중 오류가 발생했습니다.\n오류: ${e.toString()}');
      }
    }
  }

  String _buildAnalysisPrompt() {
    final buffer = StringBuffer();
    buffer.writeln('다음은 교통사고 정보입니다. 과실 비율을 분석해주세요:');
    buffer.writeln('');

    for (var i = 0; i < _questions.length; i++) {
      final question = _questions[i];
      final answer = _userAnswers[question.id];
      if (answer != null && answer.toString().isNotEmpty) {
        buffer.writeln('${i + 1}. ${question.question}');
        buffer.writeln('   답변: $answer');
        buffer.writeln('');
      }
    }

    buffer.writeln('위 정보를 바탕으로:');
    buffer.writeln('1. 예상 과실 비율 (나:상대)');
    buffer.writeln('2. 과실 비율 판단 근거');
    buffer.writeln('3. 주의사항 및 조언');
    buffer.writeln('을 한국어로 자세히 설명해주세요.');

    return buffer.toString();
  }

Future<Map<String, dynamic>> _callBackendAnalysis(String prompt) async {
    // 1. 노란 줄이 떴던 프라이빗 게터(_baseUrl)를 호출합니다.
    final String currentBaseUrl = _baseUrl;

    // 2. 주소가 비어있는지 확인하여 오류를 사전에 방지합니다.
    if (currentBaseUrl.isEmpty) {
      throw Exception('.env 파일에서 BACKEND_URL을 읽어오지 못했습니다. assets/.env 설정과 main.dart 로드 경로를 확인하세요.');
    }

    // 3. 주소 끝의 슬래시(/)를 제거한 후 /analyze를 붙여 정확한 전체 주소를 생성합니다.
    final String cleanUrl = currentBaseUrl.endsWith('/') 
        ? currentBaseUrl.substring(0, currentBaseUrl.length - 1) 
        : currentBaseUrl;
    
    final url = Uri.parse('$cleanUrl/analyze');
    
    // 디버깅을 위해 실제 요청이 가는 주소를 콘솔에 출력합니다.
    debugPrint('🚀 API 요청 주소: $url');

    // 4. Multipart 요청 생성
    final request = http.MultipartRequest('POST', url);

    // 텍스트 쿼리 추가
    request.fields['text_query'] = prompt;

    // 5. 이미지 파일 추가 (웹 환경 호환 방식)
    for (var photo in widget.accidentPhotos) {
      final bytes = await photo.readAsBytes();
      request.files.add(http.MultipartFile.fromBytes(
        'image_files',
        bytes,
        filename: photo.name,
      ));
    }

    // 6. 서버로 요청 전송 및 응답 처리
    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode == 200) {
      // 한글 깨짐 방지를 위해 utf8로 디코딩합니다.
      final responseBody = utf8.decode(response.bodyBytes);
      final data = jsonDecode(responseBody);
      return data;
    } else {
      final errorBody = utf8.decode(response.bodyBytes);
      throw Exception('API 호출 실패: ${response.statusCode}\n$errorBody');
    }
  }

  Future<void> _callBackendChat(String threadId, String userMessage) async {
    setState(() {
      _isLoading = true;
    });

    try {
      final url = Uri.parse('$_baseUrl/chat');
      final response = await http.post(
        url,
        headers: {
          'Accept': 'application/json; charset=utf-8',
        },
        body: {
          'thread_id': threadId,
          'user_message': userMessage,
        },
      );

      if (response.statusCode == 200) {
        final responseBody = utf8.decode(response.bodyBytes);
        final data = jsonDecode(responseBody);
        final aiResponse = data['response'] ?? '응답을 받지 못했습니다.';
        _addBotMessage(aiResponse);
      } else {
        final errorBody = utf8.decode(response.bodyBytes);
        _addBotMessage('챗봇 응답 오류: ${response.statusCode}\n$errorBody');
      }
    } catch (e) {
      _addBotMessage('챗봇 연결 오류: $e');
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  void _scrollToBottom() {
    Future.delayed(const Duration(milliseconds: 100), () {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  // ==================== UI ====================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFFAFAF8),
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
        title: Column(
          children: [
            const Text(
              'AI 과실 분석',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: Color(0xFF222222),
              ),
            ),
            if (!_inChatMode)
              const Text(
                '● 분석 진행중',
                style: TextStyle(
                  fontSize: 11,
                  color: Color(0xFF43A047),
                  fontWeight: FontWeight.w500,
                ),
              ),
            if (_inChatMode)
              const Text(
                '● 추가 질문 모드',
                style: TextStyle(
                  fontSize: 11,
                  color: Color(0xFF43A047),
                  fontWeight: FontWeight.w500,
                ),
              ),
          ],
        ),
        centerTitle: true,
        actions: [
          Padding(
            padding: const EdgeInsets.all(8),
            child: GestureDetector(
              onTap: () {
                setState(() {
                  _messages.clear();
                  _userAnswers.clear();
                  _currentStep = 0;
                  _chatStarted = false;
                  _inChatMode = false;
                  _currentThreadId = null;
                });
                _startChat();
              },
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
      body: SafeArea(
        child: Column(
          children: [
            // 프로그레스 바 (질문 진행 중일 때만)
            if (_currentStep > 0 && _currentStep <= _questions.length && !_inChatMode)
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 12, 20, 8),
                child: Row(
                  children: [
                    Expanded(
                      child: ClipRRect(
                        borderRadius: BorderRadius.circular(4),
                        child: LinearProgressIndicator(
                          value: _currentStep / _questions.length,
                          minHeight: 6,
                          backgroundColor: const Color(0xFFF0F0F0),
                          valueColor: const AlwaysStoppedAnimation<Color>(Color(0xFF43A047)),
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Text(
                      '$_currentStep/${_questions.length}',
                      style: const TextStyle(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                        color: Color(0xFF43A047),
                      ),
                    ),
                  ],
                ),
              ),

            // 채팅 메시지 리스트
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

            // 로딩
            if (_isLoading)
              const Padding(
                padding: EdgeInsets.all(16.0),
                child: CircularProgressIndicator(color: Color(0xFF43A047)),
              ),

            // 텍스트 입력
            if (_currentStep == _questions.length - 1 || _inChatMode)
              _buildTextInput(),
          ],
        ),
      ),
    );
  }

  Widget _buildMessageBubble(ChatMessage message) {
    bool isUser = message.isUser;

    return Column(
      crossAxisAlignment: isUser ? CrossAxisAlignment.end : CrossAxisAlignment.start,
      children: [
        // 봇 아바타 라벨
        if (!isUser)
          Padding(
            padding: const EdgeInsets.only(bottom: 4, left: 4),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: const [
                Text('🤖', style: TextStyle(fontSize: 14)),
                SizedBox(width: 4),
                Text(
                  'AI 상담사',
                  style: TextStyle(
                    fontSize: 11,
                    color: Color(0xFFAAAAAA),
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),

        // 버블
        Align(
          alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
          child: Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            constraints: BoxConstraints(
              maxWidth: MediaQuery.of(context).size.width * 0.78,
            ),
            decoration: BoxDecoration(
              color: isUser ? const Color(0xFF222222) : Colors.white,
              borderRadius: BorderRadius.only(
                topLeft: Radius.circular(isUser ? 18 : 4),
                topRight: Radius.circular(isUser ? 4 : 18),
                bottomLeft: const Radius.circular(18),
                bottomRight: const Radius.circular(18),
              ),
              border: isUser ? null : Border.all(color: const Color(0xFFF5F5F5)),
              boxShadow: [
                if (!isUser)
                  BoxShadow(
                    color: Colors.black.withOpacity(0.04),
                    blurRadius: 4,
                    offset: const Offset(0, 1),
                  ),
              ],
            ),
            child: Text(
              message.text,
              style: TextStyle(
                color: isUser ? Colors.white : const Color(0xFF333333),
                fontSize: 15,
                fontWeight: FontWeight.w500,
                height: 1.5,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildOptionsWidget(QuestionData question) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      width: MediaQuery.of(context).size.width * 0.88,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFF0F0F0)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 가이드 텍스트
          Row(
            children: const [
              Text('👇', style: TextStyle(fontSize: 13)),
              SizedBox(width: 4),
              Text(
                '하나를 선택하세요',
                style: TextStyle(
                  fontSize: 12,
                  color: Color(0xFFAAAAAA),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),

          if (question.type == QuestionType.timeAndBlackbox)
            _buildTimeAndBlackboxOptions(question),
          if (question.type == QuestionType.radio)
            _buildRadioOptions(question),
          if (question.type == QuestionType.multiSelect)
            _buildMultiSelectOptions(question),
          if (question.type == QuestionType.faultFactors)
            _buildFaultFactorsOptions(question),
        ],
      ),
    );
  }

  Widget _buildOptionChip({
    required String text,
    required VoidCallback onTap,
    bool isSelected = false,
    String? emoji,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        decoration: BoxDecoration(
          color: isSelected ? const Color(0xFFE8F5E9) : const Color(0xFFFAFAFA),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected ? const Color(0xFF66BB6A) : const Color(0xFFEEEEEE),
            width: 1.5,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (emoji != null) ...[
              Text(emoji, style: const TextStyle(fontSize: 14)),
              const SizedBox(width: 6),
            ],
            Text(
              text,
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.w600,
                color: isSelected ? const Color(0xFF2E7D32) : const Color(0xFF333333),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimeAndBlackboxOptions(QuestionData question) {
    return _buildOptionChip(
      text: '시간 및 블랙박스 설정',
      emoji: '⏰',
      onTap: () => _showCustomTimePicker(question),
    );
  }

  void _showCustomTimePicker(QuestionData question) {
    int selectedHour = TimeOfDay.now().hour;
    int selectedMinute = (TimeOfDay.now().minute ~/ 5) * 5;
    bool isAm = selectedHour < 12;
    String? selectedBlackbox;

    final hourController = FixedExtentScrollController(
      initialItem: selectedHour % 12,
    );
    final minuteController = FixedExtentScrollController(
      initialItem: selectedMinute ~/ 5,
    );

    showDialog(
      context: context,
      barrierColor: Colors.black.withOpacity(0.35),
      builder: (dialogContext) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            final displayHour = isAm
                ? (selectedHour % 12 == 0 ? 12 : selectedHour % 12)
                : (selectedHour % 12 == 0 ? 12 : selectedHour % 12);

            return Dialog(
              backgroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
              insetPadding: const EdgeInsets.symmetric(horizontal: 40),
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // 헤더
                    const Text('⏰', style: TextStyle(fontSize: 36)),
                    const SizedBox(height: 8),
                    const Text(
                      '사고 발생 시간',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        color: Color(0xFF222222),
                      ),
                    ),
                    const SizedBox(height: 4),
                    const Text(
                      '사고가 발생한 시간을 선택해주세요',
                      style: TextStyle(fontSize: 13, color: Color(0xFFAAAAAA)),
                    ),
                    const SizedBox(height: 14),

                    // 지금 시간 버튼
                    GestureDetector(
                      onTap: () {
                        final now = TimeOfDay.now();
                        setDialogState(() {
                          selectedHour = now.hour;
                          selectedMinute = (now.minute ~/ 5) * 5;
                          isAm = now.hour < 12;
                          hourController.jumpToItem(selectedHour % 12);
                          minuteController.jumpToItem(selectedMinute ~/ 5);
                        });
                      },
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        decoration: BoxDecoration(
                          color: const Color(0xFFE8F5E9),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFF66BB6A), width: 1.5),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: const [
                            Text('🕐', style: TextStyle(fontSize: 14)),
                            SizedBox(width: 6),
                            Text(
                              '지금 시간으로 선택',
                              style: TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF2E7D32),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                    const SizedBox(height: 14),

                    // 휠 피커
                    SizedBox(
                      height: 150,
                      child: Row(
                        children: [
                          // 시간 휠
                          Expanded(
                            child: Stack(
                              alignment: Alignment.center,
                              children: [
                                Container(
                                  height: 40,
                                  decoration: BoxDecoration(
                                    color: const Color(0xFFE8F5E9),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                ),
                                ListWheelScrollView.useDelegate(
                                  controller: hourController,
                                  itemExtent: 40,
                                  perspective: 0.005,
                                  diameterRatio: 1.5,
                                  physics: const FixedExtentScrollPhysics(),
                                  onSelectedItemChanged: (index) {
                                    setDialogState(() {
                                      selectedHour = isAm ? index % 12 : (index % 12) + 12;
                                    });
                                  },
                                  childDelegate: ListWheelChildBuilderDelegate(
                                    builder: (context, index) {
                                      if (index < 0 || index > 11) return null;
                                      final hour = index == 0 ? 12 : index;
                                      final isSelected = index == (selectedHour % 12);
                                      return Center(
                                        child: Text(
                                          hour.toString().padLeft(2, '0'),
                                          style: TextStyle(
                                            fontSize: isSelected ? 26 : 18,
                                            fontWeight: isSelected ? FontWeight.w800 : FontWeight.w500,
                                            color: isSelected
                                                ? const Color(0xFF222222)
                                                : const Color(0xFFCCCCCC),
                                          ),
                                        ),
                                      );
                                    },
                                    childCount: 12,
                                  ),
                                ),
                              ],
                            ),
                          ),

                          // 콜론
                          const Padding(
                            padding: EdgeInsets.symmetric(horizontal: 4),
                            child: Text(
                              ':',
                              style: TextStyle(
                                fontSize: 28,
                                fontWeight: FontWeight.w800,
                                color: Color(0xFF222222),
                              ),
                            ),
                          ),

                          // 분 휠 (5분 단위)
                          Expanded(
                            child: Stack(
                              alignment: Alignment.center,
                              children: [
                                Container(
                                  height: 40,
                                  decoration: BoxDecoration(
                                    color: const Color(0xFFE8F5E9),
                                    borderRadius: BorderRadius.circular(10),
                                  ),
                                ),
                                ListWheelScrollView.useDelegate(
                                  controller: minuteController,
                                  itemExtent: 40,
                                  perspective: 0.005,
                                  diameterRatio: 1.5,
                                  physics: const FixedExtentScrollPhysics(),
                                  onSelectedItemChanged: (index) {
                                    setDialogState(() {
                                      selectedMinute = index * 5;
                                    });
                                  },
                                  childDelegate: ListWheelChildBuilderDelegate(
                                    builder: (context, index) {
                                      if (index < 0 || index > 11) return null;
                                      final minute = index * 5;
                                      final isSelected = minute == selectedMinute;
                                      return Center(
                                        child: Text(
                                          minute.toString().padLeft(2, '0'),
                                          style: TextStyle(
                                            fontSize: isSelected ? 26 : 18,
                                            fontWeight: isSelected ? FontWeight.w800 : FontWeight.w500,
                                            color: isSelected
                                                ? const Color(0xFF222222)
                                                : const Color(0xFFCCCCCC),
                                          ),
                                        ),
                                      );
                                    },
                                    childCount: 12,
                                  ),
                                ),
                              ],
                            ),
                          ),

                          // 오전/오후 토글
                          Padding(
                            padding: const EdgeInsets.only(left: 10),
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                GestureDetector(
                                  onTap: () {
                                    setDialogState(() {
                                      isAm = true;
                                      selectedHour = selectedHour % 12;
                                    });
                                  },
                                  child: Container(
                                    width: 48,
                                    height: 38,
                                    decoration: BoxDecoration(
                                      color: isAm
                                          ? const Color(0xFF43A047)
                                          : const Color(0xFFF0F0F0),
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Center(
                                      child: Text(
                                        '오전',
                                        style: TextStyle(
                                          fontSize: 13,
                                          fontWeight: FontWeight.w700,
                                          color: isAm ? Colors.white : const Color(0xFFBBBBBB),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                                const SizedBox(height: 8),
                                GestureDetector(
                                  onTap: () {
                                    setDialogState(() {
                                      isAm = false;
                                      selectedHour = (selectedHour % 12) + 12;
                                    });
                                  },
                                  child: Container(
                                    width: 48,
                                    height: 38,
                                    decoration: BoxDecoration(
                                      color: !isAm
                                          ? const Color(0xFF43A047)
                                          : const Color(0xFFF0F0F0),
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Center(
                                      child: Text(
                                        '오후',
                                        style: TextStyle(
                                          fontSize: 13,
                                          fontWeight: FontWeight.w700,
                                          color: !isAm ? Colors.white : const Color(0xFFBBBBBB),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 20),

                    // 블랙박스 선택
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: const Color(0xFFFAFAFA),
                        borderRadius: BorderRadius.circular(14),
                        border: Border.all(color: const Color(0xFFF0F0F0)),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '📹 블랙박스 유무',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF333333),
                            ),
                          ),
                          const SizedBox(height: 10),
                          Row(
                            children: [
                              Expanded(
                                child: GestureDetector(
                                  onTap: () => setDialogState(() => selectedBlackbox = '있음'),
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(vertical: 12),
                                    decoration: BoxDecoration(
                                      color: selectedBlackbox == '있음'
                                          ? const Color(0xFFE8F5E9)
                                          : Colors.white,
                                      borderRadius: BorderRadius.circular(10),
                                      border: Border.all(
                                        color: selectedBlackbox == '있음'
                                            ? const Color(0xFF66BB6A)
                                            : const Color(0xFFEEEEEE),
                                        width: 1.5,
                                      ),
                                    ),
                                    child: Center(
                                      child: Text(
                                        '📹 있음',
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w700,
                                          color: selectedBlackbox == '있음'
                                              ? const Color(0xFF2E7D32)
                                              : const Color(0xFF666666),
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: GestureDetector(
                                  onTap: () => setDialogState(() => selectedBlackbox = '없음'),
                                  child: Container(
                                    padding: const EdgeInsets.symmetric(vertical: 12),
                                    decoration: BoxDecoration(
                                      color: selectedBlackbox == '없음'
                                          ? const Color(0xFFE8F5E9)
                                          : Colors.white,
                                      borderRadius: BorderRadius.circular(10),
                                      border: Border.all(
                                        color: selectedBlackbox == '없음'
                                            ? const Color(0xFF66BB6A)
                                            : const Color(0xFFEEEEEE),
                                        width: 1.5,
                                      ),
                                    ),
                                    child: Center(
                                      child: Text(
                                        '❌ 없음',
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontWeight: FontWeight.w700,
                                          color: selectedBlackbox == '없음'
                                              ? const Color(0xFF2E7D32)
                                              : const Color(0xFF666666),
                                        ),
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

                    const SizedBox(height: 18),

                    // 확인 버튼
                    GestureDetector(
                      onTap: () {
                        if (selectedBlackbox == null) return;
                        Navigator.pop(dialogContext);
                        final timeStr = '${selectedHour.toString().padLeft(2, '0')}:${selectedMinute.toString().padLeft(2, '0')}';
                        _handleAnswer(
                          question.id,
                          {'time': timeStr, 'blackbox': selectedBlackbox},
                          '시간: $timeStr, 블랙박스: $selectedBlackbox',
                        );
                      },
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: selectedBlackbox != null
                                ? [const Color(0xFF66BB6A), const Color(0xFF43A047)]
                                : [const Color(0xFFE0E0E0), const Color(0xFFD0D0D0)],
                          ),
                          borderRadius: BorderRadius.circular(14),
                          boxShadow: selectedBlackbox != null
                              ? [
                                  BoxShadow(
                                    color: const Color(0xFF43A047).withOpacity(0.25),
                                    blurRadius: 12,
                                    offset: const Offset(0, 4),
                                  ),
                                ]
                              : [],
                        ),
                        child: Center(
                          child: Text(
                            selectedBlackbox != null ? '확인 ✓' : '블랙박스 유무를 선택해주세요',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: selectedBlackbox != null
                                  ? Colors.white
                                  : const Color(0xFF999999),
                            ),
                          ),
                        ),
                      ),
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

  // 이모지 매핑
  String _getOptionEmoji(String option) {
    const emojiMap = {
      '교차로': '🚦', '직선도로': '🛣️', '주차장': '🅿️', '기타/모름': '❓',
      '직진': '⬆️', '좌회전': '↩️', '우회전': '↪️', '유턴': '🔄',
      '정지': '🛑', '후진': '⬇️', '주차': '🅿️', '기타': '❓',
      '앞면': '⬆️', '옆면(왼쪽)': '⬅️', '옆면(오른쪽)': '➡️', '뒷면': '⬇️',
      '과속': '💨', '신호위반': '🚦', '중앙선 침범': '⚠️',
      '안전거리 미확보': '📏', '끼어들기': '🔀', '음주운전': '🍺', '해당없음': '✅',
    };
    return emojiMap[option] ?? '•';
  }

  Widget _buildRadioOptions(QuestionData question) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: question.options.map((option) => _buildOptionChip(
        text: option,
        emoji: _getOptionEmoji(option),
        onTap: () => _handleAnswer(question.id, option, option),
      )).toList(),
    );
  }

  Widget _buildMultiSelectOptions(QuestionData question) {
    final Set<String> selected = {};
    return StatefulBuilder(builder: (context, setStateLocal) {
      return Column(
        children: [
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: question.options.map((opt) {
              return _buildOptionChip(
                text: opt,
                emoji: _getOptionEmoji(opt),
                isSelected: selected.contains(opt),
                onTap: () => setStateLocal(() {
                  selected.contains(opt) ? selected.remove(opt) : selected.add(opt);
                }),
              );
            }).toList(),
          ),
          const SizedBox(height: 14),
          GestureDetector(
            onTap: selected.isEmpty
                ? () {}
                : () => _handleAnswer(question.id, selected.toList(), selected.join(', ')),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
              decoration: BoxDecoration(
                color: selected.isEmpty ? const Color(0xFFE0E0E0) : const Color(0xFF43A047),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                '선택 완료 ✓',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w700,
                  color: selected.isEmpty ? const Color(0xFF999999) : Colors.white,
                ),
              ),
            ),
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
          const Text(
            '🚗 나의 과실',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: question.options.map((opt) => _buildOptionChip(
              text: opt,
              emoji: _getOptionEmoji(opt),
              isSelected: my.contains(opt),
              onTap: () => setStateLocal(() {
                my.contains(opt) ? my.remove(opt) : my.add(opt);
              }),
            )).toList(),
          ),
          const SizedBox(height: 16),
          const Text(
            '🚙 상대의 과실',
            style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Color(0xFF333333)),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: question.options.map((opt) => _buildOptionChip(
              text: opt,
              emoji: _getOptionEmoji(opt),
              isSelected: op.contains(opt),
              onTap: () => setStateLocal(() {
                op.contains(opt) ? op.remove(opt) : op.add(opt);
              }),
            )).toList(),
          ),
          const SizedBox(height: 16),
          Center(
            child: GestureDetector(
              onTap: () {
                final myStr = my.isEmpty ? '해당없음' : my.join(', ');
                final oppStr = op.isEmpty ? '해당없음' : op.join(', ');
                _handleAnswer(
                  question.id,
                  {'my': my.toList(), 'opponent': op.toList()},
                  '나: $myStr\n상대: $oppStr',
                );
              },
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 10),
                decoration: BoxDecoration(
                  color: const Color(0xFF43A047),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: const Text(
                  '선택 완료 ✓',
                  style: TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ),
            ),
          ),
        ],
      );
    });
  }

  Widget _buildTextInput() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: const BoxDecoration(
        color: Colors.white,
        border: Border(top: BorderSide(color: Color(0xFFF0F0F0))),
      ),
      child: Row(
        children: [
          Expanded(
            child: Container(
              height: 46,
              padding: const EdgeInsets.symmetric(horizontal: 16),
              decoration: BoxDecoration(
                color: const Color(0xFFF5F5F5),
                borderRadius: BorderRadius.circular(14),
              ),
              child: TextField(
                controller: _textController,
                style: const TextStyle(
                  fontSize: 15,
                  color: Color(0xFF333333),
                ),
                decoration: InputDecoration(
                  hintText: _inChatMode ? '추가 질문을 입력하세요...' : '추가 정보 입력 (선택사항)',
                  hintStyle: const TextStyle(
                    fontSize: 14,
                    color: Color(0xFFAAAAAA),
                  ),
                  border: InputBorder.none,
                  contentPadding: const EdgeInsets.symmetric(vertical: 12),
                ),
                maxLines: 1,
              ),
            ),
          ),
          const SizedBox(width: 10),
          GestureDetector(
            onTap: () async {
              final text = _textController.text.trim();
              _textController.clear();

              if (_inChatMode && _currentThreadId != null) {
                _addUserMessage(text.isNotEmpty ? text : '(입력 없음)');
                await _callBackendChat(_currentThreadId!, text);
              } else {
                _handleAnswer(
                  _questions[_currentStep].id,
                  text,
                  text.isNotEmpty ? text : '(입력 없음)',
                );
              }
            },
            child: Container(
              width: 46,
              height: 46,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFF66BB6A), Color(0xFF43A047)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
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
              child: const Icon(Icons.send, size: 20, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }
}

// 채팅 메시지 모델
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
    this.questionData,
  });
}

// 질문 데이터 모델
class QuestionData {
  final String id;
  final String question;
  final QuestionType type;
  final List<String> options;

  QuestionData({
    required this.id,
    required this.question,
    required this.type,
    required this.options,
  });
}

// 질문 타입
enum QuestionType {
  radio,
  multiSelect,
  timeAndBlackbox,
  faultFactors,
  text,
}