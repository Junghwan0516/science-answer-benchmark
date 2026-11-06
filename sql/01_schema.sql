PRAGMA foreign_keys = ON;
CREATE TABLE domains (domain TEXT PRIMARY KEY NOT NULL);
CREATE TABLE labels (label TEXT PRIMARY KEY NOT NULL);
CREATE TABLE questions (
 question_id TEXT PRIMARY KEY NOT NULL,
 domain TEXT NOT NULL REFERENCES domains(domain),
 question_text TEXT NOT NULL CHECK(length(trim(question_text)) > 0),
 reference_answer TEXT NOT NULL CHECK(length(trim(reference_answer)) > 0)
);
CREATE TABLE answers (
 sample_id TEXT PRIMARY KEY NOT NULL,
 answer_id TEXT NOT NULL UNIQUE,
 question_id TEXT NOT NULL REFERENCES questions(question_id),
 student_answer TEXT NOT NULL CHECK(length(trim(student_answer)) > 0),
 label TEXT NOT NULL REFERENCES labels(label),
 normalized_input_hash TEXT NOT NULL CHECK(length(normalized_input_hash) = 64)
);
CREATE TABLE answer_features (
 sample_id TEXT PRIMARY KEY NOT NULL REFERENCES answers(sample_id),
 answer_char_count INTEGER NOT NULL CHECK(answer_char_count > 0),
 answer_word_count INTEGER NOT NULL CHECK(answer_word_count > 0),
 lexical_token_count INTEGER NOT NULL CHECK(lexical_token_count >= 0),
 unique_word_count INTEGER NOT NULL CHECK(unique_word_count >= 0 AND unique_word_count <= lexical_token_count),
 type_token_ratio REAL NOT NULL CHECK(type_token_ratio BETWEEN 0 AND 1),
 question_token_jaccard REAL NOT NULL CHECK(question_token_jaccard BETWEEN 0 AND 1),
 reference_token_jaccard REAL NOT NULL CHECK(reference_token_jaccard BETWEEN 0 AND 1),
 reference_length_ratio REAL NOT NULL CHECK(reference_length_ratio > 0),
 punctuation_fraction REAL NOT NULL CHECK(punctuation_fraction BETWEEN 0 AND 1),
 has_numeric_content INTEGER NOT NULL CHECK(has_numeric_content IN (0,1)),
 raw_input_hash TEXT NOT NULL CHECK(length(raw_input_hash) = 64)
);
CREATE TABLE development_assignments (
 sample_id TEXT PRIMARY KEY NOT NULL REFERENCES answers(sample_id),
 internal_split TEXT NOT NULL CHECK(internal_split IN ('fit','development','calibration')),
 input_multiplicity INTEGER NOT NULL CHECK(input_multiplicity >= 1),
 has_conflicting_labels INTEGER NOT NULL CHECK(has_conflicting_labels IN (0,1)),
 primary_sample_weight REAL NOT NULL CHECK(primary_sample_weight = 1),
 sensitivity_fit_weight REAL,
 CHECK((internal_split = 'fit' AND sensitivity_fit_weight IS NOT NULL AND sensitivity_fit_weight > 0 AND sensitivity_fit_weight <= 1)
    OR (internal_split <> 'fit' AND sensitivity_fit_weight IS NULL))
);
CREATE INDEX answers_question_idx ON answers(question_id);
CREATE INDEX answers_label_idx ON answers(label);
CREATE INDEX answers_input_idx ON answers(normalized_input_hash);
CREATE INDEX assignments_split_idx ON development_assignments(internal_split);
