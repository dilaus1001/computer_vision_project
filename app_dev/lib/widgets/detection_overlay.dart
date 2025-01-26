import 'package:flutter/material.dart';
import '../models/detection_result.dart';

class DetectionOverlay extends StatelessWidget {
  final List<DetectionResult> detections;
  final Size imageSize;
  
  // Added optional parameters for customization
  final Color boxColor;
  final double strokeWidth;
  final double textSize;

  const DetectionOverlay({
    super.key,
    required this.detections,
    required this.imageSize,
    this.boxColor = Colors.red,
    this.strokeWidth = 2.0,
    this.textSize = 16.0,
  });

  @override
  Widget build(BuildContext context) {
    // CustomPaint widget uses our custom painter to draw detection boxes
    return CustomPaint(
      size: imageSize,
      painter: DetectionPainter(
        detections: detections,
        imageSize: imageSize,
        boxColor: boxColor,
        strokeWidth: strokeWidth,
        textSize: textSize,
      ),
    );
  }
}

class DetectionPainter extends CustomPainter {
  final List<DetectionResult> detections;
  final Size imageSize;
  final Color boxColor;
  final double strokeWidth;
  final double textSize;

  DetectionPainter({
    required this.detections,
    required this.imageSize,
    required this.boxColor,
    required this.strokeWidth,
    required this.textSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Create paint objects for different drawing needs
    final boxPaint = Paint()
      ..color = boxColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth;

    // Background paint for text
    final backgroundPaint = Paint()
      ..color = Colors.white.withOpacity(0.8)
      ..style = PaintingStyle.fill;

    for (var detection in detections) {
      // Calculate scaling factors to map detection coordinates to screen coordinates
      final double scaleX = size.width / imageSize.width;
      final double scaleY = size.height / imageSize.height;

      // Create the rect using the detection's properties
      final rect = Rect.fromLTWH(
        detection.x * scaleX,
        detection.y * scaleY,
        detection.width * scaleX,
        detection.height * scaleY,
      );

      // Draw bounding box
      canvas.drawRect(rect, boxPaint);

      // Prepare the detection label text
      final labelText = '${detection.label} ${(detection.confidence * 100).toStringAsFixed(0)}%';
      
      // Create text painter for the label
      final textPainter = TextPainter(
        text: TextSpan(
          text: labelText,
          style: TextStyle(
            color: boxColor,
            fontSize: textSize,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      
      // Layout the text
      textPainter.layout();

      // Calculate text position
      final textOffset = Offset(
        rect.left,
        rect.top - textPainter.height - 4, // Position above the box
      );

      // Draw text background
      canvas.drawRect(
        Rect.fromLTWH(
          textOffset.dx - 2,
          textOffset.dy - 2,
          textPainter.width + 4,
          textPainter.height + 4,
        ),
        backgroundPaint,
      );

      // Draw the text
      textPainter.paint(canvas, textOffset);
    }
  }

  @override
  bool shouldRepaint(DetectionPainter oldDelegate) {
    // Optimize repainting by checking if relevant properties have changed
    return oldDelegate.detections != detections ||
           oldDelegate.imageSize != imageSize ||
           oldDelegate.boxColor != boxColor ||
           oldDelegate.strokeWidth != strokeWidth ||
           oldDelegate.textSize != textSize;
  }
}