import 'package:flutter/material.dart';
import '../models/detection_result.dart';

class ResultsScreen extends StatelessWidget {
  final DetectionResult detection;
  final double malignancyProbability;
  
  const ResultsScreen({
    required this.detection,
    required this.malignancyProbability,
  });

  String _getRiskLevel(double probability) {
    if (probability < 0.2) return 'Low Risk';
    if (probability < 0.7) return 'Medium Risk';
    return 'High Risk - Please consult a doctor';
  }

  Color _getRiskColor(double probability) {
    if (probability < 0.2) return Colors.green;
    if (probability < 0.7) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Analysis Results'),
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Card(
              child: Padding(
                padding: EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Risk Assessment',
                      style: Theme.of(context).textTheme.titleLarge,
                    ),
                    SizedBox(height: 16),
                    Text(
                      _getRiskLevel(malignancyProbability),
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                        color: _getRiskColor(malignancyProbability),
                      ),
                    ),
                    SizedBox(height: 8),
                    LinearProgressIndicator(
                      value: malignancyProbability,
                      backgroundColor: Colors.grey[200],
                      valueColor: AlwaysStoppedAnimation<Color>(
                        _getRiskColor(malignancyProbability),
                      ),
                    ),
                    SizedBox(height: 16),
                    Text(
                      'Confidence Level: ${(detection.confidence * 100).toStringAsFixed(1)}%',
                      style: Theme.of(context).textTheme.titleMedium,
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 16),
            if (malignancyProbability > 0.7)
              Card(
                color: Colors.red[50],
                child: Padding(
                  padding: EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Icon(Icons.warning, color: Colors.red, size: 48),
                      SizedBox(height: 8),
                      Text(
                        'Please consult a healthcare professional',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: Colors.red,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}