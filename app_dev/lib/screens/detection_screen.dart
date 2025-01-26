import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:camera/camera.dart';
import 'package:skc_app/models/detection_result.dart';
import '../services/camera_service.dart';
import '../services/model_service.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:typed_data';

class DetectionScreen extends StatefulWidget {
  @override
  _DetectionScreenState createState() => _DetectionScreenState();
}

class _DetectionScreenState extends State<DetectionScreen> {
  final CameraService _cameraService = CameraService();
  final ModelService _modelService = ModelService();
  bool isProcessing = false;
  File? imageFile;
  bool isTestMode = false;
  
  @override
  void initState() {
    super.initState();
    _initializeServices();
  }
  
  Future<void> _initializeServices() async {
    try {
      await _cameraService.initialize();
      await _modelService.initialize();
      setState(() {});
      print('1. Service correctly initializated');
    } catch (e) {
      print('Error initializing services: $e');
    }
  }

  // Future<void> _processImage() async {
  //   setState(() => isProcessing = true);
    
  //   try {
  //     // Capture image
  //     // final image = await _cameraService.takePicture();
  //     // if (image == null) return;

  //     // Simulate taking a picture by using a placeholder image
  //         // Instead of using File directly, we'll create a temporary file from the asset
  //     final ByteData data = await rootBundle.load('assets/malignant_test_image.jpg');
  //     final Directory tempDir = await getTemporaryDirectory();
  //     final String tempPath = '${tempDir.path}/temp_image.jpg';
  //     final File tempFile = File(tempPath);
      
  //     // Write the asset to a temporary file
  //     await tempFile.writeAsBytes(
  //       data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)
  //     );


  //     // Detect moles
  //     final moles = await _modelService.detectMoles(File(image.path));
      
  //     // Analyze each detected mole
  //     for (var mole in moles) {
  //       final malignancyProbability = await _modelService.analyzeMole(
  //         File(image.path),
  //         mole,
  //       );
        
  //       // Show results (implement UI feedback)
  //       _showResults(mole, malignancyProbability);
  //     }
  //   } catch (e) {
  //     print('Error processing image: $e');
  //   } finally {
  //     setState(() => isProcessing = false);
  //   }
  // }

  // Future<void> _processImage() async {
  //   setState(() => isProcessing = true);
    
  //   try {
  //     // Capture image
  //     final image = await _cameraService.takePicture();
  //     if (image == null) return;

  //     // Simulate taking a picture by using a placeholder image
  //         // Instead of using File directly, we'll create a temporary file from the asset
  //     // final ByteData data = await rootBundle.load('assets/malignant_test_image.jpg');
  //     // final Directory tempDir = await getTemporaryDirectory();
  //     // final String tempPath = '${tempDir.path}/temp_image.jpg';
  //     // final File tempFile = File(tempPath);
      
  //     // Write the asset to a temporary file
  //     // await tempFile.writeAsBytes(
  //     //   data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)
  //     // );


  //     // Detect moles
  //     final tempDir = await getTemporaryDirectory();
  //     tempFile = File('${tempDir.path}/temp_image.jpg');
      
  //     // Write the captured image data to the temp file
  //     await tempFile!.writeAsBytes(
  //       image.buffer.asUint8List(image.offsetInBytes, image.lengthInBytes)
  //     );

  //     final moles = await _modelService.detectMoles(tempFile!);
      
  //     // Analyze each detected mole
  //     for (var mole in moles) {
  //       final malignancyProbability = await _modelService.analyzeMole(
  //         tempFile,
  //         mole,
  //       );
        
  //       // Show results (implement UI feedback)
  //       _showResults(mole, malignancyProbability);
  //     }

  //     if (await tempFile.exists()) {
  //       await tempFile.delete();
  //   }
  //   } catch (e) {
  //     print('Error processing image: $e');
  //   } finally {
  //     setState(() => isProcessing = false);
  //   }
  // }

  Future<void> _processImage() async {
    print('Start image processing ...');
    setState(() => isProcessing = true);
    
    try {
      if (isTestMode) {
        // Use test image from assets
        final ByteData data = await rootBundle.load('assets/malignant_test_image.jpg');
        final Directory tempDir = await getTemporaryDirectory();
        imageFile = File('${tempDir.path}/temp_image.jpg');
        
        await imageFile!.writeAsBytes(
          data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes)
        );
        print('image correctly loaded');
      } else {
        // Take picture from camera
        final XFile? capturedImage = await _cameraService.takePicture();
        if (capturedImage == null) return;
        imageFile = File(capturedImage.path);
      }

      // Process the image (whether from camera or test asset)
      if (imageFile != null) {
        // Detect moles
        print('Starting mole detection');
        final moles = await _modelService.detectMoles(imageFile!);
        
        // Analyze each detected mole
        print('start mole classification');
        for (var mole in moles) {
          final malignancyProbability = await _modelService.analyzeMole(
            imageFile!,
            mole,
          );
          // Show results
          _showResults(mole, malignancyProbability);
        }

        // Clean up the file after processing
        if (await imageFile!.exists()) {
          await imageFile!.delete();
        }
      }
    } catch (e) {
      print('Error processing image: $e');
    } finally {
      setState(() => isProcessing = false);
    }
  }


void _showResults(DetectionResult mole, double probability) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      title: Text('Detection Results', style: TextStyle(fontWeight: FontWeight.bold)),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            'Mole detected with ${(mole.confidence * 100).toStringAsFixed(1)}% confidence.',
            style: TextStyle(fontSize: 16),
          ),
          SizedBox(height: 8),
          Text(
            'Malignancy Probability: ${(probability * 100).toStringAsFixed(1)}%',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
        ],
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: Text('OK'),
        ),
      ],
    ),
  );
}
@override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: GestureDetector(
          onTap: () => Navigator.pop(context),
          child: Container(
            margin: EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black45,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Center(
              child: Icon(
                CupertinoIcons.back,
                color: Colors.white,
                size: 24,
              ),
            ),
          ),
        ),
        actions: [
          // Add test mode toggle
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Switch(
              value: isTestMode,
              onChanged: (value) {
                setState(() {
                  isTestMode = value;
                });
              },
            ),
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              Expanded(
                child: isTestMode 
                  ? Center(child: Text('Test Mode Active'))
                  : Container(
                      width: double.infinity,
                      child: _cameraService.controller?.buildPreview() ??
                          Center(child: CircularProgressIndicator()),
                    ),
              ),
              SafeArea(
                child: Container(
                  height: 120,
                  color: Colors.black,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      CupertinoButton(
                        padding: EdgeInsets.zero,
                        onPressed: () => Navigator.pop(context),
                        child: Text(
                          'Cancel',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                          ),
                        ),
                      ),
                      GestureDetector(
                        onTap: isProcessing ? null : _processImage,
                        child: Container(
                          width: 72,
                          height: 72,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: Colors.white,
                              width: 5,
                            ),
                          ),
                        ),
                      ),
                      SizedBox(width: 65),
                    ],
                  ),
                ),
              ),
            ],
          ),
          if (isProcessing)
            Container(
              color: Colors.black87,
              child: Center(
                child: CupertinoActivityIndicator(
                  radius: 20,
                  color: Colors.white,
                ),
              ),
            ),
        ],
      ),
    );
  }

@override
  void dispose() {
    _cameraService.dispose();
    _modelService.dispose();
    super.dispose();
  }
}