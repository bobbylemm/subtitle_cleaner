#!/bin/bash
curl -X POST "http://localhost:8080/v1/clean/" \
  -H "X-API-Key: test-api-key-123" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_srt.srt" \
  -F "format=srt" \
  -F "settings={\"remove_fillers\":true,\"fix_punctuation\":false}" \
  -s
