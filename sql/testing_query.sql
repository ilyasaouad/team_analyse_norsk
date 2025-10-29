-- First Query: Joining Tables
SELECT *
FROM tls201_appln t201
JOIN TLS207_PERS_APPLN t207 ON t201.appln_id = t207.appln_id
JOIN TLS206_PERSON t206 ON t207.person_id = t206.person_id
WHERE docdb_family_id IN (68208238, 69159669, 69172731);

-- Second Query: Selecting Distinct Sectors
SELECT DISTINCT psn_sector
FROM tls206_person
WHERE psn_sector IS NOT NULL
ORDER BY psn_sector;

-- Third Query: Getting Column Information
SELECT COLUMN_NAME,
       DATA_TYPE,
       CHARACTER_MAXIMUM_LENGTH,
       NUMERIC_PRECISION,
       NUMERIC_SCALE,
       IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'TLS211_PAT_PUBLN'
ORDER BY ORDINAL_POSITION;




SELECT t201.docdb_family_id,
    t201.appln_id,
    CONCAT(
        COALESCE(t201.appln_auth, ''),
        COALESCE(t201.appln_nr, ''),
        COALESCE(t201.appln_kind, '')
    ) AS application_number,
    t201.appln_filing_date,
    t211.appln_id,
    CONCAT(
        COALESCE(t211.publn_auth, ''),
        COALESCE(t211.publn_nr, ''),
        COALESCE(t211.publn_kind, '')
    ) AS publication_number,
    t211.publn_date,
    t201_priority.appln_auth AS priority_auth
FROM tls201_appln t201_priority
    JOIN tls204_appln_prior t204 ON t201_priority.appln_id = t204.prior_appln_id
    JOIN tls201_appln t201 ON t204.appln_id = t201.appln_id
    LEFT JOIN tls211_pat_publn t211 ON t201.appln_id = t211.appln_id
WHERE t201.docdb_family_id IN (44368432);
---------------
SELECT priority.appln_auth AS priority_auth,
    priority.docdb_family_id AS family_id,
    priority.appln_id AS priority_appln_id,
    priority.appln_auth AS priority_appln_auth,
    priority.appln_nr AS priority_appln_nr,
    priority.appln_kind AS priority_appln_kind,
    priority.appln_filing_date AS priority_appln_filing_date,
    later.docdb_family_id AS later_docdb_family_id,
    later.appln_id AS later_appln_id,
    later.appln_auth AS later_appln_auth,
    later.appln_nr AS later_appln_nr,
    later.appln_kind AS later_appln_kind,
    later.appln_filing_date AS later_appln_filing_date,
    t211_priority.appln_id AS t211_priority_appln_id,
    t211_priority.publn_auth AS t211_priority_publn_auth,
    t211_priority.publn_nr AS t211_priority_publn_nr,
    t211_priority.publn_kind AS t211_priority_publn_kind,
    t211_priority.publn_date AS t211_priority_publn_date,
    t211_later.appln_id AS t211_later_appln_id,
    t211_later.publn_auth AS t211_later_publn_auth,
    t211_later.publn_nr AS t211_later_publn_nr,
    t211_later.publn_kind AS t211_later_publn_kind,
    t211_later.publn_date AS t211_later_publn_date,
    t204.prior_appln_id AS t204_prior_appln_id,
    t204.appln_id AS t204_appln_id,
    t204.prior_appln_seq_nr AS t204_prior_appln_seq_nr
FROM tls201_appln priority
    JOIN tls204_appln_prior t204 ON priority.appln_id = t204.prior_appln_id
    JOIN tls201_appln later ON t204.appln_id = later.appln_id
    LEFT JOIN tls211_pat_publn t211_priority ON priority.appln_id = t211_priority.appln_id
    LEFT JOIN tls211_pat_publn t211_later ON later.appln_id = t211_later.appln_id
WHERE priority.docdb_family_id IN (69770749);