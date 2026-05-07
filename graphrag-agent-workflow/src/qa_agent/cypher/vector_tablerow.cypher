CALL db.index.vector.queryNodes('tablerow_embedding', $limit, $query_vector)
YIELD node, score
WITH node AS r, score
OPTIONAL MATCH (s:Section)-[:CONTAINS_TABLE_ROW]->(r)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
RETURN
  r.id            AS node_id,
  'TableRow'      AS node_label,
  r.indexed_text  AS indexed_text,
  r.text          AS raw_text,
  p.url           AS url,
  s.anchor        AS anchor,
  s.breadcrumb    AS breadcrumb,
  s.breadcrumb    AS title,
  score           AS vector_score
ORDER BY score DESC
LIMIT $limit
