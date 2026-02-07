import 'package:flutter/material.dart';
import 'widgets/chat_widget.dart';
import 'chat.dart'; // AccidentStep을 사용하기 위해 import

class ChatWidgetScreen extends StatelessWidget {
  final List<dynamic> accidentData;
  final String threadId; // threadId 필드 추가

  const ChatWidgetScreen({
    super.key, 
    required this.accidentData, 
    required this.threadId, // 생성자에 threadId 추가
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AI 추가 질문'),
      ),
      body: ChatWidget(
        accidentData: accidentData, 
        threadId: threadId, // ChatWidget에 threadId 전달
      ),
    );
  }
}