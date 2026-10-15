/*
 * Licensed under the Apache License, Version 2.0.
 * Adapted from castorini/anserini anserini-0.12.0
 * src/main/java/io/anserini/index/generator/DefaultLuceneDocumentGenerator.java.
 * Retain empty contents so title-only documents remain searchable. Nonempty
 * documents delegate unchanged to Anserini's original implementation.
 */
package io.anserini.index.generator;

import io.anserini.collection.InvalidContentsException;
import io.anserini.collection.MultifieldSourceDocument;
import io.anserini.collection.SourceDocument;
import io.anserini.index.IndexArgs;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.util.BytesRef;

public class BeirMultifieldGenerator<T extends SourceDocument>
    extends DefaultLuceneDocumentGenerator<T> {
  public BeirMultifieldGenerator(IndexArgs args) { super(args); }

  @Override
  public Document createDocument(T src) throws GeneratorException {
    final String contents;
    try { contents = src.contents(); }
    catch (InvalidContentsException e) { throw new InvalidDocumentException(); }
    if (!contents.trim().isEmpty()) return super.createDocument(src);
    Document document = new Document();
    document.add(new StringField(IndexArgs.ID, src.id(), Field.Store.YES));
    document.add(new SortedDocValuesField(IndexArgs.ID, new BytesRef(src.id())));
    if (args.storeRaw) document.add(new StoredField(IndexArgs.RAW, src.raw()));
    FieldType fieldType = new FieldType();
    fieldType.setStored(args.storeContents);
    if (args.storeDocvectors) {
      fieldType.setStoreTermVectors(true);
      fieldType.setStoreTermVectorPositions(true);
    }
    fieldType.setIndexOptions(args.storePositions
        ? IndexOptions.DOCS_AND_FREQS_AND_POSITIONS : IndexOptions.DOCS_AND_FREQS);
    document.add(new Field(IndexArgs.CONTENTS, contents, fieldType));
    if (src instanceof MultifieldSourceDocument) {
      ((MultifieldSourceDocument) src).fields().forEach((k, v) -> {
        if (IndexArgs.ENTITY.equals(k)) document.add(new StoredField(k, v));
        else document.add(new Field(k, v, fieldType));
      });
    }
    return document;
  }
}
