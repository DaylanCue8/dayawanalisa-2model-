import 'dart:typed_data';
import 'package:crop_your_image/crop_your_image.dart';
import 'package:flutter/material.dart';

class ImageCropperScreen extends StatefulWidget {
  final Uint8List imageData;

  const ImageCropperScreen({super.key, required this.imageData});

  @override
  State<ImageCropperScreen> createState() => _ImageCropperScreenState();
}

class _ImageCropperScreenState extends State<ImageCropperScreen> {
  final CropController controller = CropController();
  bool _isCropping = false;

  void _onCropped(Uint8List croppedData) {
    Navigator.of(context).pop(croppedData);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        elevation: 0,
        title: const Text('Adjust Border', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: const Icon(Icons.close, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          TextButton(
            onPressed: _isCropping
                ? null
                : () {
                    setState(() => _isCropping = true);
                    controller.crop();
                  },
            child: const Text('Apply', style: TextStyle(color: Colors.white, fontSize: 16)),
          ),
        ],
      ),
      body: SafeArea(
        child: Stack(
          children: [
            Crop(
              image: widget.imageData,
              controller: controller,
              onCropped: _onCropped,
              withCircleUi: false,
              baseColor: Colors.black,
              maskColor: Colors.black.withOpacity(0.4),
              cornerDotBuilder: (size, edgeAlignment) => Container(
                width: size,
                height: size,
                decoration: const BoxDecoration(
                  color: Colors.white,
                  shape: BoxShape.circle,
                ),
              ),
            ),
            if (_isCropping)
              const Positioned.fill(
                child: ColoredBox(
                  color: Colors.black54,
                  child: Center(child: CircularProgressIndicator(color: Colors.white)),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
