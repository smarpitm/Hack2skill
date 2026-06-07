from src.preprocessing import clean_text, extract_skills, extract_experience_years, normalize_education, normalize_location

# Test 1: Clean text
print(clean_text("  Python!!!   Developer  "))  # Should print: "python developer"

# Test 2: Extract skills
text = "I know Python, Django, React.js, and AWS"
print(extract_skills(text))  # Should print: ["aws", "django", "python", "react"]

# Test 3: Extract experience
print(extract_experience_years("I have 5+ years of experience in software development"))  # Should print: 5.0

# Test 4: Normalize education
print(normalize_education("B.Tech from IIT Bombay"))  # Should print: (3, "b.tech", 1)

# Test 5: Normalize location
print(normalize_location("Bengaluru"))  # Should print: "bangalore"
