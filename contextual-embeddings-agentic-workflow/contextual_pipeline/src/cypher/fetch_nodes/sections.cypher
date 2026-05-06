MATCH (s:Section)
WHERE s.id IN $ids
OPTIONAL MATCH (parent:Section)-[:HAS_SUBSECTION]->(s)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(root:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH s, parent, collect(DISTINCT p)[0] AS p
RETURN
  s.id            AS node_id,
  s.text          AS raw_text,
  s.breadcrumb    AS breadcrumb,
  toString(coalesce(s.synthesized, false)) AS synthesized,
  parent.text     AS parent_text,
  p.title         AS page_title,
  p.description   AS page_description
