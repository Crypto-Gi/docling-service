#!/bin/bash
# Comprehensive API Test Script for Docling Service
# Tests all endpoints and validates responses

# Don't exit on error - we want to run all tests
set +e

BASE_URL="http://localhost:5010"
TEST_PDF="test_document.pdf"
RESULTS_FILE="api_test_results.txt"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Initialize results file
echo "=== Docling Service API Test Results ===" > $RESULTS_FILE
echo "Test Date: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to print and log
log_test() {
    echo -e "$1"
    echo "$2" >> $RESULTS_FILE
}

# Test 1: Health Check
echo -e "${YELLOW}Test 1: Health Check Endpoint${NC}"
HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" $BASE_URL/healthz)
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)
BODY=$(echo "$HEALTH_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" == "200" ] && echo "$BODY" | grep -q '"status".*"ok"'; then
    log_test "${GREEN}✓ PASSED${NC}" "Test 1: Health Check - PASSED"
    ((TESTS_PASSED++))
else
    log_test "${RED}✗ FAILED${NC}" "Test 1: Health Check - FAILED (HTTP $HTTP_CODE)"
    ((TESTS_FAILED++))
fi
echo "" >> $RESULTS_FILE

# Test 2: Cloud Storage Status
echo -e "${YELLOW}Test 2: Cloud Storage Status Endpoint${NC}"
CLOUD_RESPONSE=$(curl -s -w "\n%{http_code}" $BASE_URL/api/cloud-storage/status)
HTTP_CODE=$(echo "$CLOUD_RESPONSE" | tail -n1)
BODY=$(echo "$CLOUD_RESPONSE" | head -n-1)

if [ "$HTTP_CODE" == "200" ] && echo "$BODY" | grep -q '"enabled"'; then
    log_test "${GREEN}✓ PASSED${NC}" "Test 2: Cloud Storage Status - PASSED"
    echo "  Response: $BODY" >> $RESULTS_FILE
    ((TESTS_PASSED++))
else
    log_test "${RED}✗ FAILED${NC}" "Test 2: Cloud Storage Status - FAILED (HTTP $HTTP_CODE)"
    ((TESTS_FAILED++))
fi
echo "" >> $RESULTS_FILE

# Test 3: Submit PDF Conversion
echo -e "${YELLOW}Test 3: Submit PDF Conversion${NC}"
if [ ! -f "$TEST_PDF" ]; then
    log_test "${RED}✗ FAILED${NC}" "Test 3: PDF Conversion - FAILED (Test PDF not found)"
    ((TESTS_FAILED++))
else
    CONVERT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST $BASE_URL/api/convert -F "file=@$TEST_PDF")
    HTTP_CODE=$(echo "$CONVERT_RESPONSE" | tail -n1)
    BODY=$(echo "$CONVERT_RESPONSE" | head -n-1)
    
    if [ "$HTTP_CODE" == "202" ] && echo "$BODY" | grep -q '"task_id"'; then
        TASK_ID=$(echo "$BODY" | jq -r '.task_id')
        log_test "${GREEN}✓ PASSED${NC}" "Test 3: PDF Conversion Submission - PASSED"
        echo "  Task ID: $TASK_ID" >> $RESULTS_FILE
        ((TESTS_PASSED++))
        
        # Test 4: Poll Status
        echo -e "${YELLOW}Test 4: Poll Conversion Status${NC}"
        sleep 2
        STATUS_RESPONSE=$(curl -s -w "\n%{http_code}" $BASE_URL/api/status/$TASK_ID)
        HTTP_CODE=$(echo "$STATUS_RESPONSE" | tail -n1)
        BODY=$(echo "$STATUS_RESPONSE" | head -n-1)
        
        if [ "$HTTP_CODE" == "200" ] && echo "$BODY" | grep -q '"status"'; then
            STATUS=$(echo "$BODY" | jq -r '.status')
            log_test "${GREEN}✓ PASSED${NC}" "Test 4: Status Polling - PASSED"
            echo "  Current Status: $STATUS" >> $RESULTS_FILE
            ((TESTS_PASSED++))
            
            # Test 5: Wait for Completion and Download
            echo -e "${YELLOW}Test 5: Wait for Completion & Download Result${NC}"
            echo "  Waiting for conversion to complete..."
            
            MAX_WAIT=120  # 2 minutes
            WAITED=0
            while [ $WAITED -lt $MAX_WAIT ]; do
                STATUS_RESPONSE=$(curl -s $BASE_URL/api/status/$TASK_ID)
                STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status')
                
                if [ "$STATUS" == "completed" ]; then
                    log_test "${GREEN}✓ PASSED${NC}" "Test 5: Conversion Completed - PASSED"
                    echo "  Conversion time: ${WAITED}s" >> $RESULTS_FILE
                    ((TESTS_PASSED++))
                    
                    # Test 6: Download Markdown
                    echo -e "${YELLOW}Test 6: Download Markdown Result${NC}"
                    DOWNLOAD_RESPONSE=$(curl -s -w "\n%{http_code}" -o test_result.md $BASE_URL/api/result/$TASK_ID)
                    HTTP_CODE=$(echo "$DOWNLOAD_RESPONSE" | tail -n1)
                    
                    if [ "$HTTP_CODE" == "200" ] && [ -f "test_result.md" ]; then
                        FILE_SIZE=$(wc -c < test_result.md)
                        log_test "${GREEN}✓ PASSED${NC}" "Test 6: Download Markdown - PASSED"
                        echo "  File size: $FILE_SIZE bytes" >> $RESULTS_FILE
                        echo "  First 200 chars:" >> $RESULTS_FILE
                        head -c 200 test_result.md >> $RESULTS_FILE
                        echo "" >> $RESULTS_FILE
                        ((TESTS_PASSED++))
                    else
                        log_test "${RED}✗ FAILED${NC}" "Test 6: Download Markdown - FAILED (HTTP $HTTP_CODE)"
                        ((TESTS_FAILED++))
                    fi
                    
                    # Test 7: Get JSON Result
                    echo -e "${YELLOW}Test 7: Get JSON Result${NC}"
                    JSON_RESPONSE=$(curl -s -w "\n%{http_code}" $BASE_URL/api/result/$TASK_ID/json)
                    HTTP_CODE=$(echo "$JSON_RESPONSE" | tail -n1)
                    BODY=$(echo "$JSON_RESPONSE" | head -n-1)
                    
                    if [ "$HTTP_CODE" == "200" ] && echo "$BODY" | jq -e '.markdown_content' > /dev/null 2>&1; then
                        MARKDOWN_LENGTH=$(echo "$BODY" | jq -r '.markdown_content' | wc -c)
                        MARKDOWN_URL=$(echo "$BODY" | jq -r '.markdown_url // "N/A"')
                        log_test "${GREEN}✓ PASSED${NC}" "Test 7: JSON Result - PASSED"
                        echo "  Markdown length: $MARKDOWN_LENGTH chars" >> $RESULTS_FILE
                        echo "  Cloud URL: $MARKDOWN_URL" >> $RESULTS_FILE
                        ((TESTS_PASSED++))
                    else
                        log_test "${RED}✗ FAILED${NC}" "Test 7: JSON Result - FAILED (HTTP $HTTP_CODE)"
                        ((TESTS_FAILED++))
                    fi
                    
                    break
                elif [ "$STATUS" == "failed" ]; then
                    DETAIL=$(echo "$STATUS_RESPONSE" | jq -r '.detail // "Unknown error"')
                    log_test "${RED}✗ FAILED${NC}" "Test 5: Conversion Failed - $DETAIL"
                    ((TESTS_FAILED++))
                    TESTS_FAILED=$((TESTS_FAILED + 2))  # Skip tests 6 and 7
                    break
                fi
                
                sleep 3
                WAITED=$((WAITED + 3))
                echo -n "."
            done
            
            if [ $WAITED -ge $MAX_WAIT ]; then
                log_test "${RED}✗ FAILED${NC}" "Test 5: Conversion Timeout - exceeded ${MAX_WAIT}s"
                ((TESTS_FAILED++))
                TESTS_FAILED=$((TESTS_FAILED + 2))  # Skip tests 6 and 7
            fi
            echo ""
        else
            log_test "${RED}✗ FAILED${NC}" "Test 4: Status Polling - FAILED (HTTP $HTTP_CODE)"
            ((TESTS_FAILED++))
            TESTS_FAILED=$((TESTS_FAILED + 3))  # Skip tests 5, 6, and 7
        fi
    else
        log_test "${RED}✗ FAILED${NC}" "Test 3: PDF Conversion Submission - FAILED (HTTP $HTTP_CODE)"
        ((TESTS_FAILED++))
        TESTS_FAILED=$((TESTS_FAILED + 4))  # Skip tests 4, 5, 6, and 7
    fi
fi
echo "" >> $RESULTS_FILE

# Test 8: Invalid File Format
echo -e "${YELLOW}Test 8: Invalid File Format Rejection${NC}"
echo "test" > test_invalid.txt
INVALID_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST $BASE_URL/api/convert -F "file=@test_invalid.txt")
HTTP_CODE=$(echo "$INVALID_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" == "400" ]; then
    log_test "${GREEN}✓ PASSED${NC}" "Test 8: Invalid Format Rejection - PASSED"
    ((TESTS_PASSED++))
else
    log_test "${RED}✗ FAILED${NC}" "Test 8: Invalid Format Rejection - FAILED (Expected 400, got $HTTP_CODE)"
    ((TESTS_FAILED++))
fi
rm -f test_invalid.txt
echo "" >> $RESULTS_FILE

# Test 9: Missing File/URL
echo -e "${YELLOW}Test 9: Missing File/URL Validation${NC}"
MISSING_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST $BASE_URL/api/convert)
HTTP_CODE=$(echo "$MISSING_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" == "400" ]; then
    log_test "${GREEN}✓ PASSED${NC}" "Test 9: Missing File/URL Validation - PASSED"
    ((TESTS_PASSED++))
else
    log_test "${RED}✗ FAILED${NC}" "Test 9: Missing File/URL Validation - FAILED (Expected 400, got $HTTP_CODE)"
    ((TESTS_FAILED++))
fi
echo "" >> $RESULTS_FILE

# Test 10: Non-existent Task ID
echo -e "${YELLOW}Test 10: Non-existent Task ID Handling${NC}"
NOTFOUND_RESPONSE=$(curl -s -w "\n%{http_code}" $BASE_URL/api/status/nonexistent123)
HTTP_CODE=$(echo "$NOTFOUND_RESPONSE" | tail -n1)

if [ "$HTTP_CODE" == "404" ]; then
    log_test "${GREEN}✓ PASSED${NC}" "Test 10: Non-existent Task ID - PASSED"
    ((TESTS_PASSED++))
else
    log_test "${RED}✗ FAILED${NC}" "Test 10: Non-existent Task ID - FAILED (Expected 404, got $HTTP_CODE)"
    ((TESTS_FAILED++))
fi
echo "" >> $RESULTS_FILE

# Summary
echo -e "\n${YELLOW}=== Test Summary ===${NC}"
echo "" >> $RESULTS_FILE
echo "=== Test Summary ===" >> $RESULTS_FILE
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo "Tests Passed: $TESTS_PASSED" >> $RESULTS_FILE
echo "Tests Failed: $TESTS_FAILED" >> $RESULTS_FILE

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
SUCCESS_RATE=$((TESTS_PASSED * 100 / TOTAL_TESTS))
echo -e "${YELLOW}Success Rate: $SUCCESS_RATE%${NC}"
echo "Success Rate: $SUCCESS_RATE%" >> $RESULTS_FILE

echo ""
echo "Full results saved to: $RESULTS_FILE"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Check $RESULTS_FILE for details.${NC}"
    exit 1
fi
