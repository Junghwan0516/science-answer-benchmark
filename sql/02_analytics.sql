CREATE VIEW v_question_metrics AS
SELECT q.question_id, q.domain, COUNT(*) AS answer_count,
       SUM(a.label = 'correct') AS correct_count,
       1.0 * SUM(a.label = 'correct') / COUNT(*) AS observed_correct_share,
       AVG(f.answer_word_count) AS mean_answer_words,
       AVG(f.reference_token_jaccard) AS mean_reference_overlap
FROM questions q JOIN answers a USING(question_id)
JOIN answer_features f USING(sample_id)
GROUP BY q.question_id, q.domain;

CREATE VIEW v_domain_summary AS
SELECT q.domain, COUNT(*) AS answer_count,
       COUNT(DISTINCT q.question_id) AS question_count,
       SUM(a.label = 'correct') AS correct_count,
       1.0 * SUM(a.label = 'correct') / COUNT(*) AS observed_correct_share,
       AVG(f.answer_word_count) AS mean_answer_words
FROM questions q JOIN answers a USING(question_id)
JOIN answer_features f USING(sample_id)
GROUP BY q.domain;

CREATE VIEW v_label_summary AS
SELECT a.label, COUNT(*) AS answer_count,
       COUNT(DISTINCT a.question_id) AS question_count,
       AVG(f.answer_word_count) AS mean_answer_words,
       AVG(f.reference_token_jaccard) AS mean_reference_overlap
FROM answers a JOIN answer_features f USING(sample_id)
GROUP BY a.label;

CREATE VIEW v_partition_summary AS
SELECT p.internal_split, COUNT(*) AS answer_count,
       COUNT(DISTINCT a.question_id) AS question_count,
       SUM(a.label = 'non_domain') AS non_domain_count
FROM development_assignments p JOIN answers a USING(sample_id)
GROUP BY p.internal_split;
