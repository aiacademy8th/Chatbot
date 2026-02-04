import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'result.dart';  
import 'dart:io';
import 'package:flutter_dotenv/flutter_dotenv.dart';


// bool _chatStarted = false;

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
  
  // OpenAI API 키 (실제 키로 교체 필요!)
  // final openaiApiKey = '';
  final openaiApiKey = dotenv.env['OPENAI_API_KEY'];
  
  // 채팅 메시지 리스트
  final List<ChatMessage> _messages = [];
  
  // 현재 단계 (0-6)
  int _currentStep = 0;
  
  // 사용자 응답 저장
  final Map<String, dynamic> _userAnswers = {};
  
  // 로딩 상태
  bool _isLoading = false;

  // 7단계 질문 정의
  final List<QuestionData> _questions = [
    // 1. 시간, 블랙박스 유무
    QuestionData(
      id: 'time_blackbox',
      question: '사고 발생 시간과 블랙박스 유무를 선택해주세요.',
      type: QuestionType.timeAndBlackbox,
      options: ['있음', '없음'],
    ),
    
    // 2. 장소 유형
    QuestionData(
      id: 'location_type',
      question: '사고 장소는 어디인가요?',
      type: QuestionType.radio,
      options: ['교차로', '직선도로', '주차장', '기타/모름'],
    ),
    
    // 3. 나의 상태
    QuestionData(
      id: 'my_action',
      question: '사고 당시 나의 주행 상태는?',
      type: QuestionType.radio,
      options: ['직진', '좌회전', '우회전', '유턴', '정지', '후진', '주차', '기타'],
    ),
    
    // 4. 상대 상태
    QuestionData(
      id: 'opponent_action',
      question: '사고 당시 상대방의 주행 상태는?',
      type: QuestionType.radio,
      options: ['직진', '좌회전', '우회전', '유턴', '정지', '후진', '주차', '기타'],
    ),
    
    // 5. 충돌 부위
    QuestionData(
      id: 'collision_part',
      question: '내 차량의 어느 부위가 충돌했나요? (중복 선택 가능)',
      type: QuestionType.multiSelect,
      options: ['앞면', '옆면(왼쪽)', '옆면(오른쪽)', '뒷면'],
    ),
    
    // 6. 과실 요소
    QuestionData(
      id: 'fault_factors',
      question: '과실 요소를 선택해주세요. (중복 선택 가능)',
      type: QuestionType.faultFactors,
      options: ['과속', '신호위반', '중앙선 침범', '안전거리 미확보', '끼어들기', '음주운전', '해당없음'],
    ),
    
    // 7. 추가 정보
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

  final String? openaiApiKey = dotenv.env['OPENAI_API_KEY'];
  print('-------------------------');
  print('API 키 확인: $openaiApiKey');
  print('-------------------------');

  
  // 딱 한 번만 실행되도록 보장
  WidgetsBinding.instance.addPostFrameCallback((_) {
    // if (_messages.isEmpty) {
    _startChat();
    // }
  });
}

void _startChat() {
    if (_chatStarted) return;
    _chatStarted = true;
  _addBotMessage('안녕하세요! 🚗\n교통사고 과실 비율 분석을 도와드리겠습니다.');
  
  Future.delayed(const Duration(milliseconds: 600), () {
    // 현재 단계가 0이고 메시지가 너무 많지 않을 때만 질문
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

  // 현재 단계의 질문 표시
  void _askCurrentQuestion() {
    if (_currentStep < _questions.length) {
      final question = _questions[_currentStep];
      _addBotMessage(question.question);
      
      // 선택지 버튼 추가
      if (question.type != QuestionType.text) {
        _addOptionsMessage(question);
      }
    } else {
      // 모든 질문 완료 - GPT 분석 요청
      _analyzeWithGPT();
    }
  }

  // 봇 메시지 추가
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

  // 사용자 메시지 추가
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

  // 선택지 메시지 추가
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

  // 답변 처리
  void _handleAnswer(String questionId, dynamic answer, String displayText) {
    // 답변 저장
    _userAnswers[questionId] = answer;
    
    // 사용자 메시지로 표시
    _addUserMessage(displayText);
    
    // 다음 단계로
    _currentStep++;
    
    // 잠시 후 다음 질문
    Future.delayed(const Duration(milliseconds: 500), () {
      _askCurrentQuestion();
    });
  }

  // // GPT로 분석 요청
  // Future<void> _analyzeWithGPT() async {
  //   _addBotMessage('입력하신 정보를 분석 중입니다... 🤔');
    
  //   setState(() {
  //     _isLoading = true;
  //   });

  //   try {
  //     final prompt = _buildAnalysisPrompt();
  //     final response = await _callGPT(prompt);
      
  //     _addBotMessage(response);
  //   } catch (e) {
  //     _addBotMessage('죄송합니다. 분석 중 오류가 발생했습니다.\n오류: $e');
  //   } finally {
  //     setState(() {
  //       _isLoading = false;
  //     });
  //   }
  // }

  Future<void> _analyzeWithGPT() async {
  _addBotMessage('입력하신 정보를 분석 중입니다... 🤔');
  
  if (!mounted) return;  // ⭐ 추가
  
  setState(() {
    _isLoading = true;
  });

  try {
    final prompt = _buildAnalysisPrompt();  // ⭐ 추가 (prompt 생성)
    final response = await _callGPT(prompt);
    
    if (mounted) {
      setState(() {
        _isLoading = false;  // ⭐ 로딩 종료
      });
      
      // ⭐ 결과 화면으로 이동
      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultScreen(
            analysisResult: {
              'fault_ratio': _extractFaultRatio(response),
              'analysis': response,
              'references': _getRagReferences(),  // 나중에 구현
            },
            userAnswers: _userAnswers,
          ),
        ),
      );
    }
  } catch (e) {
    if (mounted) {
      setState(() {
        _isLoading = false;
      });
      _addBotMessage('죄송합니다. 분석 중 오류가 발생했습니다.\n오류: $e');
    }
  }
}

// ⭐ 과실 비율 추출 함수 추가
String _extractFaultRatio(String gptResponse) {
  // GPT 응답에서 "30:70" 형식 찾기
  final regex = RegExp(r'(\d+)\s*:\s*(\d+)');
  final match = regex.firstMatch(gptResponse);
  
  if (match != null) {
    return '${match.group(1)}:${match.group(2)}';
  }
  
  return '50:50';  // 기본값
}

// ⭐ RAG 참고 자료 (임시 - 나중에 백엔드에서 가져옴)
List<Map<String, String>> _getRagReferences() {
  // 임시 데이터 (나중에 Python 백엔드에서 받아올 예정)
  return [
    {
      'source': '도로교통법 제14조',
      'content': '교차로에서는 우선 순위를 지켜야 합니다.',
    },
    {
      'source': '과실비율 인정 기준',
      'content': '신호위반 시 80% 이상 과실이 인정됩니다.',
    },
  ];
}

    


  // GPT 프롬프트 생성
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

  // OpenAI API 호출
  Future<String> _callGPT(String prompt) async {
    final url = Uri.parse('https://api.openai.com/v1/chat/completions');
    
    final response = await http.post(
      url,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $openaiApiKey',
      },
      body: jsonEncode({
        'model': 'gpt-4o-mini',
        'messages': [
          {
            'role': 'system',
            'content': '당신은 교통사고 과실 비율 분석 전문가입니다. 한국 도로교통법을 기준으로 정확하고 상세한 분석을 제공합니다.',
          },
          {
            'role': 'user',
            'content': prompt,
          },
        ],
        'max_tokens': 1000,
        'temperature': 0.7,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(utf8.decode(response.bodyBytes));
      return data['choices'][0]['message']['content'];
    } else {
      throw Exception('API 호출 실패: ${response.statusCode}\n${response.body}');
    }
  }

  // 스크롤 맨 아래로
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI 과실 비율 분석'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              // 다시 시작
              setState(() {
                _messages.clear();
                _userAnswers.clear();
                _currentStep = 0;
                _chatStarted = false;
              });
              _addBotMessage('새로 시작합니다. 😊');
              _startChat();
              // Future.delayed(const Duration(milliseconds: 500), () {
              //   _askCurrentQuestion();
              // });
            },
          ),
        ],
      ),
      body: Column(
        children: [
          // 진행 상황 표시
          if (_currentStep < _questions.length)
            LinearProgressIndicator(
              value: _currentStep / _questions.length,
              backgroundColor: Colors.grey.shade200,
              valueColor: AlwaysStoppedAnimation<Color>(Colors.blue.shade400),
            ),
          
          // 채팅 메시지 리스트
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(16),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                
                if (message.isOptions) {
                  return _buildOptionsWidget(message.questionData!);
                }
                
                return _buildMessageBubble(message);
              },
            ),
          ),
          
          // 로딩 인디케이터
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          
          // 텍스트 입력 (7번째 질문용)
          if (_currentStep == 6)
            _buildTextInput(),
        ],
      ),
    );
  }

  // 메시지 버블
  Widget _buildMessageBubble(ChatMessage message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.75,
        ),
        decoration: BoxDecoration(
          color: message.isUser ? Colors.blue.shade500 : Colors.grey.shade200,
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          message.text,
          style: TextStyle(
            color: message.isUser ? Colors.white : Colors.black87,
            fontSize: 15,
          ),
        ),
      ),
    );
  }

  // 선택지 위젯
  Widget _buildOptionsWidget(QuestionData question) {
    if (question.type == QuestionType.timeAndBlackbox) {
      return _buildTimeAndBlackboxOptions(question);
    } else if (question.type == QuestionType.multiSelect) {
      return _buildMultiSelectOptions(question);
    } else if (question.type == QuestionType.faultFactors) {
      return _buildFaultFactorsOptions(question);
    } else {
      return _buildRadioOptions(question);
    }
  }

  // 시간 + 블랙박스 선택
  Widget _buildTimeAndBlackboxOptions(QuestionData question) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('사고 시간 / 블랙박스 유무:', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            ElevatedButton.icon(
              icon: const Icon(Icons.access_time),
              label: const Text('시간 선택'),
              onPressed: () async {
                final TimeOfDay? picked = await showTimePicker(
                  context: context,
                  initialTime: TimeOfDay.now(),
                );
                if (picked != null) {
                  final timeStr = '${picked.hour}:${picked.minute.toString().padLeft(2, '0')}';
                  
                  // 블랙박스 선택
                  showDialog(
                    context: context,
                    builder: (context) => AlertDialog(
                      title: const Text('블랙박스 유무'),
                      content: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ListTile(
                            title: const Text('있음'),
                            onTap: () {
                              Navigator.pop(context);
                              _handleAnswer(
                                question.id,
                                {'time': timeStr, 'blackbox': '있음'},
                                '시간: $timeStr, 블랙박스: 있음',
                              );
                            },
                          ),
                          ListTile(
                            title: const Text('없음'),
                            onTap: () {
                              Navigator.pop(context);
                              _handleAnswer(
                                question.id,
                                {'time': timeStr, 'blackbox': '없음'},
                                '시간: $timeStr, 블랙박스: 없음',
                              );
                            },
                          ),
                        ],
                      ),
                    ),
                  );
                }
              },
            ),
          ],
        ),
      ),
    );
  }

  // 라디오 버튼 선택
  Widget _buildRadioOptions(QuestionData question) {
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(8),
        child: Wrap(
          spacing: 8,
          runSpacing: 8,
          children: question.options.map((option) {
            return ChoiceChip(
              label: Text(option),
              selected: false,
              onSelected: (selected) {
                if (selected) {
                  _handleAnswer(question.id, option, option);
                }
              },
            );
          }).toList(),
        ),
      ),
    );
  }

  // 다중 선택 (충돌 부위)
  Widget _buildMultiSelectOptions(QuestionData question) {
    final Set<String> selectedOptions = {};
    
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: StatefulBuilder(
        builder: (context, setLocalState) {
          return Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: question.options.map((option) {
                    final isSelected = selectedOptions.contains(option);
                    return FilterChip(
                      label: Text(option),
                      selected: isSelected,
                      onSelected: (selected) {
                        setLocalState(() {
                          if (selected) {
                            selectedOptions.add(option);
                          } else {
                            selectedOptions.remove(option);
                          }
                        });
                      },
                    );
                  }).toList(),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: selectedOptions.isEmpty
                      ? null
                      : () {
                          _handleAnswer(
                            question.id,
                            selectedOptions.toList(),
                            selectedOptions.join(', '),
                          );
                        },
                  child: const Text('선택 완료'),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  // 과실 요소 (나/상대 구분)
  Widget _buildFaultFactorsOptions(QuestionData question) {
    final Set<String> myFaults = {};
    final Set<String> opponentFaults = {};
    
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: StatefulBuilder(
        builder: (context, setLocalState) {
          return Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('나의 과실 요소:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: question.options.map((option) {
                    final isSelected = myFaults.contains(option);
                    return FilterChip(
                      label: Text(option),
                      selected: isSelected,
                      selectedColor: Colors.red.shade200,
                      onSelected: (selected) {
                        setLocalState(() {
                          if (selected) {
                            myFaults.add(option);
                          } else {
                            myFaults.remove(option);
                          }
                        });
                      },
                    );
                  }).toList(),
                ),
                const SizedBox(height: 16),
                const Text('상대의 과실 요소:', style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: question.options.map((option) {
                    final isSelected = opponentFaults.contains(option);
                    return FilterChip(
                      label: Text(option),
                      selected: isSelected,
                      selectedColor: Colors.blue.shade200,
                      onSelected: (selected) {
                        setLocalState(() {
                          if (selected) {
                            opponentFaults.add(option);
                          } else {
                            opponentFaults.remove(option);
                          }
                        });
                      },
                    );
                  }).toList(),
                ),
                const SizedBox(height: 12),
                ElevatedButton(
                  onPressed: () {
                    final myStr = myFaults.isEmpty ? '해당없음' : myFaults.join(', ');
                    final oppStr = opponentFaults.isEmpty ? '해당없음' : opponentFaults.join(', ');
                    _handleAnswer(
                      question.id,
                      {'my': myFaults.toList(), 'opponent': opponentFaults.toList()},
                      '나: $myStr\n상대: $oppStr',
                    );
                  },
                  child: const Text('선택 완료'),
                ),
              ],
            ),
          );
        },
      ),
    );
  }

  // 텍스트 입력 필드 (7번째 질문)
  Widget _buildTextInput() {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade300,
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _textController,
              decoration: const InputDecoration(
                hintText: '추가 정보 입력 (선택사항)',
                border: OutlineInputBorder(),
              ),
              maxLines: null,
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            icon: const Icon(Icons.send),
            onPressed: () {
              final text = _textController.text.trim();
              _handleAnswer(
                _questions[_currentStep].id,
                text.isEmpty ? '없음' : text,
                text.isEmpty ? '추가 정보 없음' : text,
              );
              _textController.clear();
            },
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
  radio,              // 단일 선택
  multiSelect,        // 다중 선택
  timeAndBlackbox,    // 시간 + 블랙박스
  faultFactors,       // 과실 요소 (나/상대)
  text,               // 텍스트 입력
}