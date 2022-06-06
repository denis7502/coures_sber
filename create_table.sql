CREATE TABLE dataset
(
    Id SERIAL PRIMARY KEY,
    Points FLOAT ,
    DateCreate timestamp,
    Vector FLOAT ARRAY,
    Dest BOOLEAN,
    Path TEXT
);