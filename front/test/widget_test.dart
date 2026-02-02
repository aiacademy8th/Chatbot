import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
// import 'package:front/main.dart';

void main() {
  testWidgets('App starts correctly', (WidgetTester tester) async {
    // await tester.pumpWidget(const AccidentHelperApp());
    
    expect(find.text('사고 과실 도우미'), findsOneWidget);
    expect(find.text('119'), findsOneWidget);
    expect(find.text('112'), findsOneWidget);
    expect(find.text('보험사 연결'), findsOneWidget);
  });
}