import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../chat.dart';

class ChatWidget extends StatefulWidget {
  final List<dynamic> accidentData;
  final String threadId;

  const ChatWidget({
    super.key, 
    required this.accidentData,
    required this.threadId,
  });

  @override
  State<ChatWidget> createState() => _ChatWidgetState();
}

class _ChatWidgetState extends State<ChatWidget> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, String>> _messages = [];

  bool _isSending = false;

  String _buildPrompt(String userInput) {
    // 사고 정보와 함께 모델에 보낼 “프롬프트”를 만들어준다.
    final sb = StringBuffer();
    sb.writeln("사고 정보:");
    // for (var step in widget.accidentData) {
    //   sb.writeln("${step.title}: ${step.selectedOptions.join(', ')}");
    // }
    sb.writeln("\n질문:");
    sb.writeln(userInput);
    return sb.toString();
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;

    setState(() {
      _messages.add({"role": "user", "content": text});
      _isSending = true;
    });
    _controller.clear();

    // final prompt = _buildPrompt(text); // 프롬프트 빌드는 더 이상 필요 없음
    await _callBackendChatStream(text);

    setState(() {
      _isSending = false;
    });
  }

  Future<void> _callBackendChatStream(String userMessage) async {
    final url = Uri.parse('http://localhost:8001/chat/stream');
    
    try {
      final request = http.Request('POST', url)
        ..bodyFields = {
          'thread_id': widget.threadId,
          'user_message': userMessage,
        };

      final streamedResponse = await request.send();

      if (streamedResponse.statusCode == 200) {
        String currentBotResponse = "";
        _messages.add({"role": "bot", "content": ""}); // 스트리밍을 위한 초기 빈 메시지 추가

        await for (var chunk in streamedResponse.stream.transform(utf8.decoder).transform(const LineSplitter())) {
          if (chunk.startsWith('data: ')) {
            final jsonData = jsonDecode(chunk.substring(5)); // 'data: ' 접두사 제거
            
            if (jsonData.containsKey('chunk')) {
              currentBotResponse += jsonData['chunk'];
              setState(() {
                _messages.last["content"] = currentBotResponse;
              });
            } else if (jsonData.containsKey('done') && jsonData['done'] == true) {
              print("[Chat Stream] 스트리밍 완료");
              break; // 스트리밍 종료
            } else if (jsonData.containsKey('error')) {
              print('❌ [Backend Chat Stream Error] ${jsonData['error']}');
              currentBotResponse += "\n오류: ${jsonData['error']}";
              setState(() {
                _messages.last["content"] = currentBotResponse;
              });
              break;
            }
          }
        }
      } else {
        final errorBody = await streamedResponse.stream.bytesToString();
        print('❌ [Backend Chat Error] Status: ${streamedResponse.statusCode}, Body: $errorBody');
        setState(() {
          _messages.add({"role": "bot", "content": '오류: 백엔드와 통신에 실패했습니다. (코드: ${streamedResponse.statusCode})'});
        });
      }
    } catch (e) {
      print('❌ [Frontend Chat Error] Exception: $e');
      setState(() {
        _messages.add({"role": "bot", "content": '오류: 메시지를 보내는 중 예외가 발생했습니다.'});
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final msg = _messages[index];
              final isUser = msg["role"] == "user";
              return Align(
                alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                child: Container(
                  margin: const EdgeInsets.symmetric(vertical: 4, horizontal: 8),
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: isUser ? Colors.blue.shade100 : Colors.grey.shade200,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(msg["content"] ?? ""),
                ),
              );
            },
          ),
        ),
        if (_isSending) const LinearProgressIndicator(),
        
        // 텍스트 입력 UI (chat.dart 스타일 적용)
        Container(
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
                  controller: _controller,
                  decoration: const InputDecoration(
                    hintText: '질문을 입력하세요...',
                    border: OutlineInputBorder(),
                  ),
                  onSubmitted: (_) => _isSending ? null : _sendMessage(),
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: const Icon(Icons.send),
                onPressed: _isSending ? null : _sendMessage,
              ),
            ],
          ),
        ),
      ],
    );
  }
}
