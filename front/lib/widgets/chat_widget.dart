import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import '../chat.dart';

class ChatWidget extends StatefulWidget {
  final List<AccidentStep> accidentData;
  const ChatWidget({super.key, required this.accidentData});

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
    for (var step in widget.accidentData) {
      sb.writeln("${step.title}: ${step.selectedOptions.join(', ')}");
    }
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

    final prompt = _buildPrompt(text);
    final responseText = await _callOpenAI(prompt);

    setState(() {
      _messages.add({"role": "bot", "content": responseText});
      _isSending = false;
    });
  }

  Future<String> _callOpenAI(String prompt) async {
    const apiUrl = "https://api.openai.com/v1/chat/completions";

    const model = "gpt-4o-mini"; // 요즘 가장 가성비 좋은 챗 모델 :contentReference[oaicite:2]{index=2}

    const openAiKey = "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"; // *** 여기에 네 API 키 넣어 ***

    final response = await http.post(
      Uri.parse(apiUrl),
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer $openAiKey",
      },
      body: jsonEncode({
        "model": model,
        "messages": [
          {"role": "system", "content": "친절한 사고 상담 챗봇입니다."},
          {"role": "user", "content": prompt},
        ],
        "max_tokens": 500,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      final content = data["choices"][0]["message"]["content"];
      return content ?? "응답이 없습니다.";
    } else {
      return "오류: ${response.statusCode} ${response.body}";
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
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _controller,
                decoration: const InputDecoration(hintText: "질문을 입력하세요"),
              ),
            ),
            IconButton(
              icon: const Icon(Icons.send),
              onPressed: _isSending ? null : _sendMessage,
            )
          ],
        ),
      ],
    );
  }
}
