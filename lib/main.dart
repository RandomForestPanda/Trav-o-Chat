/*

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Chatbot',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ChatbotScreen(),
    );
  }
}

class ChatbotScreen extends StatefulWidget {
  @override
  _ChatbotScreenState createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  TextEditingController _controller = TextEditingController();
  String _responseMessage = "";
  bool _isLoading = false;

  // Function to send the query to the FastAPI server
  Future<void> sendQuery(String query) async {
    setState(() {
      _isLoading = true;
    });

    final url = Uri.parse('http://localhost:8000/query');  // FastAPI server URL
    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'query': query}),
      );

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);
        setState(() {
          _responseMessage = responseData['response'];
        });
      } else {
        setState(() {
          _responseMessage = 'Failed to get a response from the server.';
        });
      }
    } catch (error) {
      setState(() {
        _responseMessage = 'Error occurred: $error';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Solmelu Chatbot'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: <Widget>[
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: 'Ask a question',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                final query = _controller.text;
                if (query.isNotEmpty) {
                  sendQuery(query);
                }
              },
              child: _isLoading
                  ? CircularProgressIndicator(): Text('Send Query'),
            ),
            SizedBox(height: 20),
            if (_responseMessage.isNotEmpty)
              Expanded(
                child: SingleChildScrollView(
                  child: Text(
                    _responseMessage,  // Display the response string
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

*/


/* user state v2->rollback to above code
 */

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Chatbot',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ChatbotScreen(),
    );
  }
}

class ChatbotScreen extends StatefulWidget {
  @override
  _ChatbotScreenState createState() => _ChatbotScreenState();
}

class _ChatbotScreenState extends State<ChatbotScreen> {
  TextEditingController _controller = TextEditingController();
  String _responseMessage = "";
  bool _isLoading = false;

  // Function to send the query to the FastAPI server
  Future<void> sendQuery(String query) async {
    setState(() {
      _isLoading = true;
    });

    final url = Uri.parse('http://localhost:8000/query'); // Update with server IP if hosted
    final userId = "12345"; // Replace with a proper user ID system (e.g., UUID)

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'user_id': userId, 'query': query}),
      );

      if (response.statusCode == 200) {
        final responseData = json.decode(response.body);
        setState(() {
          _responseMessage = responseData['response'];
        });
      } else {
        setState(() {
          _responseMessage = 'Failed to get a response from the server.';
        });
      }
    } catch (error) {
      setState(() {
        _responseMessage = 'Error occurred: $error';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Travel Guide Chatbot'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: <Widget>[
            TextField(
              controller: _controller,
              decoration: InputDecoration(
                labelText: 'Ask a question',
                border: OutlineInputBorder(),
              ),
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                final query = _controller.text;
                if (query.isNotEmpty) {
                  sendQuery(query);
                }
              },
              child: _isLoading
                  ? CircularProgressIndicator()
                  : Text('Send Query'),
            ),
            SizedBox(height: 20),
            if (_responseMessage.isNotEmpty)
              Expanded(
                child: SingleChildScrollView(
                  child: Text(
                    _responseMessage,
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

