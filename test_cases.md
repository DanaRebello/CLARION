# Authentication Module Test Cases

## Test Case 1: Register Candidate
Input: Valid name, email, password, role=candidate  
Expected Result: User stored in database

## Test Case 2: Register Recruiter
Input: Valid details, role=recruiter  
Expected Result: User stored in database

## Test Case 3: Duplicate Email Registration
Input: Existing email  
Expected Result: Error message shown

## Test Case 4: Login with Correct Credentials
Expected Result: Redirect to respective dashboard

## Test Case 5: Login with Wrong Password
Expected Result: Login denied
