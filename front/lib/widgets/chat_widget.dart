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
    final responseText = await _callBackendChat(text);

    setState(() {
      _messages.add({"role": "bot", "content": responseText});
      _isSending = false;
    });
  }

  Future<String> _callBackendChat(String userMessage) async {
    final url = Uri.parse('http://localhost:8001/chat');
    
    try {
      final response = await http.post(
        url,
        body: {
          'thread_id': widget.threadId,
          'user_message': userMessage,
        },
      );

      if (response.statusCode == 200) {
        // UTF-8로 응답을 디코딩하여 한글 깨짐 방지
        final responseBody = utf8.decode(response.bodyBytes);
        final data = jsonDecode(responseBody);
        return data['response'] ?? '백엔드로부터 응답이 없습니다.';
      } else {
        // UTF-8로 오류 메시지 디코딩
        final errorBody = utf8.decode(response.bodyBytes);
        print('❌ [Backend Chat Error] Status: ${response.statusCode}, Body: $errorBody');
        return '오류: 백엔드와 통신에 실패했습니다. (코드: ${response.statusCode})';
      }
    } catch (e) {
      print('❌ [Frontend Chat Error] Exception: $e');
      return '오류: 메시지를 보내는 중 예외가 발생했습니다.';
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
