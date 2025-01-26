import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/detection_provider.dart';
import 'screens/detection_screen.dart';

void main() {
  // Ensure Flutter is initialized
  WidgetsFlutterBinding.ensureInitialized();
  
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => DetectionProvider()),
      ],
      child: MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Skin Cancer Detector',
      theme: ThemeData(
        primarySwatch: Colors.teal,
        brightness: Brightness.light,
        cardTheme: CardTheme(
          elevation: 4,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        textTheme: TextTheme(
          titleLarge: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
          bodyMedium: TextStyle(fontSize: 16),
        ),
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.teal,
          elevation: 0,
        ),
      ),
      home: WelcomeScreen(),
    );
  }
}


class WelcomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Welcome'),
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Text(
            //   'Welcome! What would you like to do?',
            //   style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            //   textAlign: TextAlign.center,
            // ),
            SizedBox(height: 32),
            SizedBox(
              height: 300, // Sets button height
              child: ElevatedButton(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (context) => DetectionScreen()),
                  );
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.teal, // Button color
                  foregroundColor: Colors.white, // Text color
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30), // No rounded corners
                  ),
                ),
                child: Text(
                  'New Detection',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            ),
            SizedBox(height: 16),
            SizedBox(
              height: 300, // Ensures consistent button height
              child: ElevatedButton(
                onPressed: () {
                  // Add logic to navigate to past detections
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.orange, // Button color
                  foregroundColor: Colors.white, // Text color
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(30), // No rounded corners
                  ),
                ),
                child: Text(
                  'Past Detections',
                  style: TextStyle(fontSize: 18),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}


