UNWIND $seeds AS seed
MATCH (e)
WHERE (e.id = seed.node_id OR e.name = seed.node_id OR e.key = seed.node_id)
  AND any(lbl IN labels(e)
          WHERE lbl IN ['Tool','Hook','SettingKey','PermissionMode',
                        'Provider','MessageType'])
MATCH (parent:Section)-[r:DEFINES|MENTIONS]->(e)
MATCH path = (parent)-[:HAS_SUBSECTION*0..]->(s:Section)
MATCH (s)-[:CONTAINS_CODE]->(cb:CodeBlock)
WHERE ($language IS NULL OR cb.language = $language)
OPTIONAL MATCH (p:Page)-[:HAS_SECTION]->(:Section)-[:HAS_SUBSECTION*0..]->(s)
WITH seed.node_id AS seed_id, cb, s, p,
     CASE type(r) WHEN 'DEFINES' THEN 2 ELSE 1 END AS rel_rank,
     length(path) AS section_depth,
     COALESCE(cb.position, 999) AS cb_position
RETURN
  cb.id            AS node_id,
  'CodeBlock'      AS node_label,
  cb.indexed_text  AS indexed_text,
  cb.text          AS raw_text,
  p.url            AS url,
  s.anchor         AS anchor,
  s.breadcrumb     AS breadcrumb,
  COALESCE('[' + cb.language + '] ' + s.breadcrumb, s.breadcrumb) AS title,
  seed_id          AS seed_id
ORDER BY rel_rank DESC, section_depth ASC, cb_position ASC, cb.id ASC
LIMIT $cap
