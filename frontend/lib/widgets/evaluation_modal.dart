import 'package:flutter/material.dart';

class EvaluationModal extends StatefulWidget {
  final List<dynamic> detections;
  final double averageConfidence;
  final String translatedText;
  final int sessionId;

  const EvaluationModal({
    super.key,
    required this.detections,
    required this.averageConfidence,
    required this.translatedText,
    required this.sessionId,
  });

  @override
  State<EvaluationModal> createState() => _EvaluationModalState();
}

class _EvaluationModalState extends State<EvaluationModal> {
  /// --- FILTER LOGIC (STRICT 23% THRESHOLD) ---
  /// Only detections >= 23% are shown in the UI and included in the result text.
  List<Map<String, dynamic>> get filteredDetections {
    return widget.detections
        .where((d) {
          if (d is! Map) return false;
          final conf = (d['confidence'] as num).toDouble();
          return conf >= 23.0; // STRICT: 23% and above only
        })
        .map((d) => Map<String, dynamic>.from(d as Map))
        .toList();
  }

  /// --- REBUILD RESULT TEXT FROM FILTERED DETECTIONS ---
  /// Reassembles the translated text using only characters that pass the 23% filter.
  String get filteredResultText {
    final list = filteredDetections;
    if (list.isEmpty) return "—";
    return list.map((d) => d['char']?.toString() ?? '').join('');
  }

  @override
  Widget build(BuildContext context) {
    final detectionsToShow = filteredDetections;

    return Container(
      constraints: BoxConstraints(
        maxHeight: MediaQuery.of(context).size.height * 0.85,
      ),
      decoration: const BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.vertical(top: Radius.circular(25)),
      ),
      padding: const EdgeInsets.fromLTRB(20, 10, 20, 20),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 40,
            height: 5,
            margin: const EdgeInsets.only(bottom: 10),
            decoration: BoxDecoration(
              color: Colors.grey[300],
              borderRadius: BorderRadius.circular(10),
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Center(
                    child: Text(
                      "dayaw",
                      style: TextStyle(
                        fontSize: 32,
                        fontWeight: FontWeight.bold,
                        color: Colors.orange[700],
                      ),
                    ),
                  ),
                  const SizedBox(height: 20),

                  const Text(
                    "Translation Result",
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 6),

                  Text(
                    filteredResultText,
                    style: const TextStyle(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: Colors.brown,
                    ),
                  ),
                  const SizedBox(height: 20),

                  _buildStatRow(
                    "Detected Characters:",
                    "${detectionsToShow.length}",
                  ),

                  const Divider(height: 40),

                  detectionsToShow.isEmpty
                      ? const Center(
                          child: Padding(
                            padding: EdgeInsets.all(20),
                            child: Text(
                              "No characters detected.",
                              textAlign: TextAlign.center,
                              style: TextStyle(color: Colors.grey),
                            ),
                          ),
                        )
                      : Table(
                          defaultVerticalAlignment:
                              TableCellVerticalAlignment.middle,
                          children: [
                            const TableRow(
                              decoration: BoxDecoration(
                                border: Border(
                                  bottom: BorderSide(
                                    color: Colors.grey,
                                    width: 0.5,
                                  ),
                                ),
                              ),
                              children: [
                                Padding(
                                  padding: EdgeInsets.all(8.0),
                                  child: Text(
                                    "Char",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold),
                                  ),
                                ),
                                Padding(
                                  padding: EdgeInsets.all(8.0),
                                  child: Text(
                                    "Status",
                                    style:
                                        TextStyle(fontWeight: FontWeight.bold),
                                  ),
                                ),
                              ],
                            ),
                            ...detectionsToShow.map((d) => _buildTableRow(d)),
                          ],
                        ),

                  const SizedBox(height: 20),
                  _buildTempCropStatus(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  TableRow _buildTableRow(Map<String, dynamic> d) {
    final double conf = (d['confidence'] as num).toDouble();
    final bool isExcellent = conf >= 90.0;

    return TableRow(
      children: [
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            d['char']?.toString() ?? '?',
            style: const TextStyle(fontSize: 18),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(8.0),
          child: Text(
            isExcellent ? "Excellent" : "Detected",
            style: TextStyle(
              color: isExcellent ? Colors.green : Colors.blueGrey,
              fontWeight: FontWeight.bold,
              fontSize: 11,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStatRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        children: [
          Text(label),
          const SizedBox(width: 10),
          Text(value, style: const TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildTempCropStatus() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(10),
      ),
      child: const Text(
        "Processed characters are kept in the temporary crop queue for review.",
        style: TextStyle(fontSize: 12),
      ),
    );
  }
}