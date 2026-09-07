import 'package:flutter/material.dart';
import 'screens/dayaw_landing_screen.dart'; // Import the file you just created

void main() {
  runApp(const DayawApp());
}

class DayawApp extends StatelessWidget {
  const DayawApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Dayaw',
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: Colors.brown,
      ),
      home: const DayawLandingScreen(), // The entry screen
    );
  }
}
