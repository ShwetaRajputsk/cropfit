import 'package:flutter/material.dart';
import 'main.dart'; // Import the file containing CropDiseaseHome

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}
class _SplashScreenState extends State<SplashScreen> {
  @override
  void initState() {
    super.initState();
    _navigateToHome();
  }
  _navigateToHome() async {
    await Future.delayed(Duration(seconds: 10), () {}); // Delay for 3 seconds
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(builder: (context) => CropDiseaseHome()), // Navigate to home
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.asset(
              'assets/Logo.png',
              width: 150,
              height: 150,
            ), // Logo image
            SizedBox(height: 20),
            IconButton(
              icon: Icon(Icons.arrow_forward, size: 50), // Arrow icon
              onPressed: () {
                Navigator.pushReplacement(
                  context,
                  MaterialPageRoute(builder: (context) => CropDiseaseHome()), // Navigate to home when pressed
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
