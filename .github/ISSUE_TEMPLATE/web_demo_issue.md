---
name: Web Demo Issue
about: Report issues related to the web interface
title: '[WEB] '
labels: 'web-demo'
assignees: ''

---

## Issue Type
- [ ] Web interface not loading
- [ ] Camera/microphone access problem
- [ ] WebSocket connection issue
- [ ] Visualization not working
- [ ] File upload problem
- [ ] Performance issue
- [ ] Other (please specify)

## Browser Information
- Browser: [e.g. Chrome, Firefox, Safari, Edge]
- Version: [e.g. 91.0.4472.124]
- Operating System: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]

## Server Information
- Server OS: [e.g. Ubuntu 20.04]
- Python Version: [e.g. 3.8.10]
- FastAPI/Uvicorn Version: [e.g. 0.68.0]
- How was the server started: [e.g. `python start_web_demo.py`, `uvicorn deployment.serve:app`]

## Issue Description
A clear and concise description of the issue.

## Steps to Reproduce
1. Start the web server with: `python start_web_demo.py`
2. Open browser and navigate to: http://localhost:8000
3. [Additional steps...]
4. Observe the issue

## Expected Behavior
What should happen instead?

## Actual Behavior
What actually happens?

## Browser Console Errors
Please open browser Developer Tools (F12) and paste any error messages from the Console tab:

```
Paste console errors here
```

## Network Tab Information
If there are connection issues, please check the Network tab in Developer Tools and paste relevant failed requests:

```
Failed Request: GET http://localhost:8000/ws/detection
Status: 404
Response: Not Found
```

## Server Logs
Please paste relevant server logs:

```
Paste server log output here
```

## Screenshots
If applicable, add screenshots to help explain the problem.

## File Upload Details
If the issue is related to file upload, please provide:

- File type: [e.g. .wav, .jpg, .json]
- File size: [e.g. 2.5MB]
- File format details: [e.g. 16kHz WAV, 1920x1080 JPEG]

## Device Information
If using mobile or tablet:
- Device: [e.g. iPhone 12, Samsung Galaxy S21]
- Screen Resolution: [e.g. 1920x1080]
- Touch/click behavior: [describe any touch-specific issues]

## WebRTC Permissions
Have you granted camera and microphone permissions?
- [ ] Yes, permissions were granted
- [ ] No, permissions were denied
- [ ] Permissions dialog did not appear
- [ ] Not applicable

## Additional Context
Add any other context about the problem here.

## Checklist
- [ ] I have checked the browser console for errors
- [ ] I have verified the server is running and accessible
- [ ] I have tested with the latest version
- [ ] I have included relevant logs and error messages