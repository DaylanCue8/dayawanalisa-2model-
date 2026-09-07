import 'dart:async';
import 'package:flutter/material.dart';
import '../services/api_service.dart';

/// Handles the "Tagalog to Baybayin" mode: debounced auto-translate as the
/// user types, with confidence display. Fully self-contained — owns its
/// own state, independent of the image-translation mode.
class TagalogToBaybayinView extends StatefulWidget {
  const TagalogToBaybayinView({super.key});

  @override
  State<TagalogToBaybayinView> createState() => _TagalogToBaybayinViewState();
}

class _TagalogToBaybayinViewState extends State<TagalogToBaybayinView> {
  final ApiService _apiService = ApiService();
  final TextEditingController _textController = TextEditingController();

  // Debounce timer so we don't fire a request on every keystroke — waits
  // for a short pause in typing before translating automatically.
  Timer? _debounce;

  String _translatedResult = "Result will appear here";
  double _confidenceScore = 0.0;
  bool _isLoading = false;

  @override
  void dispose() {
    _debounce?.cancel();
    _textController.dispose();
    super.dispose();
  }

  /// Called on every keystroke. Restarts a 350ms timer each time so the
  /// actual translation only fires once typing pauses.
  void _onTextChanged(String text) {
    if (_debounce?.isActive ?? false) _debounce!.cancel();
    _debounce = Timer(const Duration(milliseconds: 350), () {
      _handleTextTranslation(text);
    });
  }

  Future<void> _handleTextTranslation(String text) async {
    if (text.trim().isEmpty) {
      setState(() {
        _translatedResult = "Result will appear here";
        _confidenceScore = 0.0;
        _isLoading = false;
      });
      return;
    }

    setState(() {
      _isLoading = true;
    });

    final response = await _apiService.uploadAndTranslateDetailed(
      null,
      'Tagalog to Baybayin',
      text: text,
    );

    if (!mounted) return;

    setState(() {
      _isLoading = false;
      if (response != null) {
        _translatedResult = response['translated_text'] ?? "No result";
        _confidenceScore = (response['confidence'] as num).toDouble();
      } else {
        _translatedResult = "Error: Connection Failed";
        _confidenceScore = 0.0;
      }
    });
  }

  Color _getConfidenceColor() {
    if (_confidenceScore >= 95) return Colors.green;
    if (_confidenceScore >= 75) return Colors.orange;
    return Colors.red;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F5F5),
        borderRadius: BorderRadius.circular(15),
      ),
      clipBehavior: Clip.antiAlias,
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              child: TextField(
                controller: _textController,
                onChanged: _onTextChanged,
                maxLines: null,
                keyboardType: TextInputType.multiline,
                decoration: const InputDecoration(
                  hintText: "Enter Tagalog text here...",
                  border: InputBorder.none,
                ),
                style: const TextStyle(fontSize: 16),
              ),
            ),
            const Divider(height: 20, color: Colors.grey),
            Expanded(
              child: Align(
                alignment: Alignment.topLeft,
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          Row(
                            children: [
                              const Text(
                                "Baybayin Translation:",
                                style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey),
                              ),
                              if (_isLoading) ...[
                                const SizedBox(width: 8),
                                const SizedBox(
                                  width: 12,
                                  height: 12,
                                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.brown),
                                ),
                              ],
                            ],
                          ),
                          if (_confidenceScore > 0.0)
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                              decoration: BoxDecoration(
                                color: _getConfidenceColor().withOpacity(0.2),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                "Confidence: ${_confidenceScore.toStringAsFixed(1)}%",
                                style: TextStyle(color: _getConfidenceColor(), fontWeight: FontWeight.bold, fontSize: 12),
                              ),
                            ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _translatedResult,
                        style: const TextStyle(fontFamily: 'BaybayinCustom', fontSize: 38, color: Colors.black87, height: 2.5, letterSpacing: 6.0),
                      ),
                    ],
                  ),
                ),
              ),
            ),
            // "Translate" button removed — translation runs automatically
            // as the user types. Only a clear button remains.
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                IconButton(
                  icon: const Icon(Icons.clear, color: Colors.grey),
                  onPressed: () {
                    _debounce?.cancel();
                    _textController.clear();
                    _handleTextTranslation("");
                  },
                  tooltip: "Clear",
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}