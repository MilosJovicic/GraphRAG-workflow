MATCH (c:CodeBlock)
WHERE c.id IN $ids
OPTIONAL MATCH (s:Section)-[:CONTAINS_CODE]->(c)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH c, s, collect(DISTINCT p)[0] AS p
OPTIONAL MATCH (c)-[:EQUIVALENT_TO]-(eq:CodeBlock)
WITH c, s, p,
     [language IN collect(DISTINCT eq.language) WHERE language IS NOT NULL] AS related_hints
RETURN
  c.id                    AS node_id,
  c.text                  AS raw_text,
  c.language              AS language,
  c.preceding_paragraph   AS preceding,
  s.breadcrumb            AS breadcrumb,
  s.text                  AS parent_text,
  p.title                 AS page_title,
  p.description           AS page_description,
  related_hints           AS related_hints
