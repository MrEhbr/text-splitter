/*!
# [`MarkdownSplitter`]
Semantic splitting of Markdown documents. Tries to use as many semantic units from Markdown
as possible, according to the Common Mark specification.
*/

use std::{iter::once, ops::Range};

use either::Either;
use itertools::Itertools;
use pulldown_cmark::{Event, Options, Parser, Tag, TagEnd};

use crate::{
    splitter::{SemanticLevel, Splitter},
    trim::Trim,
    ChunkConfig, ChunkSizer,
};

use super::{ChunkCharIndex, TextChunks, TextChunksWithCharIndices};

/// Markdown splitter. Recursively splits chunks into the largest
/// semantic units that fit within the chunk size. Also will
/// attempt to merge neighboring chunks if they can fit within the
/// given chunk size.
#[derive(Debug)]
pub struct MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Method of determining chunk sizes.
    chunk_config: ChunkConfig<Sizer>,
}

impl<Sizer> MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    /// Creates a new [`MarkdownSplitter`].
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// // By default, the chunk sizer is based on characters.
    /// let splitter = MarkdownSplitter::new(512);
    /// ```
    #[must_use]
    pub fn new(chunk_config: impl Into<ChunkConfig<Sizer>>) -> Self {
        Self {
            chunk_config: chunk_config.into(),
        }
    }

    /// Generate a list of chunks from a given text. Each chunk will be up to
    /// the `max_chunk_size`.
    ///
    /// ## Method
    ///
    /// To preserve as much semantic meaning within a chunk as possible, each chunk is composed of the largest semantic units that can fit in the next given chunk. For each splitter type, there is a defined set of semantic levels. Here is an example of the steps used:
    ///
    /// 1. Characters
    /// 2. [Unicode Grapheme Cluster Boundaries](https://www.unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries)
    /// 3. [Unicode Word Boundaries](https://www.unicode.org/reports/tr29/#Word_Boundaries)
    /// 4. [Unicode Sentence Boundaries](https://www.unicode.org/reports/tr29/#Sentence_Boundaries)
    /// 5. Soft line breaks (single newline) which isn't necessarily a new element in Markdown.
    /// 6. Inline elements such as: text nodes, emphasis, strong, strikethrough, link, image, table cells, inline code, footnote references, task list markers, and inline html.
    /// 7. Block elements suce as: paragraphs, code blocks, footnote definitions, metadata. Also, a block quote or row/item within a table or list that can contain other "block" type elements, and a list or table that contains items.
    /// 8. Thematic breaks or horizontal rules.
    /// 9. Headings by level
    ///
    /// Splitting doesn't occur below the character level, otherwise you could get partial bytes of a char, which may not be a valid unicode str.
    ///
    /// Markdown is parsed according to the Commonmark spec, along with some optional features such as GitHub Flavored Markdown.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunks(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec!["# Header", "from a", "document"], chunks);
    /// ```
    pub fn chunks<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = &'text str> + 'splitter {
        Splitter::<_>::chunks(self, text)
    }

    /// Returns an iterator over chunks of the text and their byte offsets.
    /// Each chunk will be up to the `max_chunk_size`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// ```
    /// use text_splitter::MarkdownSplitter;
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![(0, "# Header"), (10, "from a"), (17, "document")], chunks);
    /// ```
    pub fn chunk_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str)> + 'splitter {
        Splitter::<_>::chunk_indices(self, text)
    }

    /// Returns an iterator over chunks of the text with their byte and character offsets.
    /// Each chunk will be up to the `chunk_capacity`.
    ///
    /// See [`MarkdownSplitter::chunks`] for more information.
    ///
    /// This will be more expensive than just byte offsets, and for most usage in Rust, just
    /// having byte offsets is sufficient. But when interfacing with other languages or systems
    /// that require character offsets, this will track the character offsets for you,
    /// accounting for any trimming that may have occurred.
    ///
    /// ```
    /// use text_splitter::{ChunkCharIndex, MarkdownSplitter};
    ///
    /// let splitter = MarkdownSplitter::new(10);
    /// let text = "# Header\n\nfrom a\ndocument";
    /// let chunks = splitter.chunk_char_indices(text).collect::<Vec<_>>();
    ///
    /// assert_eq!(vec![ChunkCharIndex { chunk: "# Header", byte_offset: 0, char_offset: 0 }, ChunkCharIndex { chunk: "from a", byte_offset: 10, char_offset: 10 }, ChunkCharIndex { chunk: "document", byte_offset: 17, char_offset: 17 }], chunks);
    /// ```
    pub fn chunk_char_indices<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = ChunkCharIndex<'text>> + 'splitter {
        Splitter::<_>::chunk_char_indices(self, text)
    }

    /// Returns an iterator over chunks along with the most recent heading at
    /// each level that precedes (or starts at) each chunk's byte offset.
    ///
    /// The [`HeadingContext`] tracks the current heading hierarchy: when a new
    /// heading is encountered at level `L`, it replaces the previous entry at
    /// `L` and clears all entries at deeper levels.
    ///
    /// Useful for contextual chunk headers in RAG pipelines — see
    /// <https://github.com/benbrandt/text-splitter/issues/116>.
    ///
    /// ```
    /// use text_splitter::{HeadingLevel, MarkdownSplitter};
    ///
    /// // Small capacity forces text-splitter to emit a chunk for each section.
    /// let splitter = MarkdownSplitter::new(15);
    /// let text = "# A\n\nintro\n\n## B\n\nbody\n";
    /// let chunks: Vec<_> = splitter.chunks_with_context(text).collect();
    /// // The last chunk sits past both headings, so its context exposes both.
    /// let (_chunk, ctx) = chunks.last().unwrap();
    /// assert_eq!(ctx.at(HeadingLevel::H1), Some("A"));
    /// assert_eq!(ctx.at(HeadingLevel::H2), Some("B"));
    /// ```
    pub fn chunks_with_context<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (&'text str, HeadingContext<'text>)> + 'splitter {
        self.chunk_indices_with_context(text)
            .map(|(_, chunk, ctx)| (chunk, ctx))
    }

    /// Like [`chunk_indices`] but additionally yields a [`HeadingContext`]
    /// reflecting the heading hierarchy active at the chunk's byte offset.
    ///
    /// See [`MarkdownSplitter::chunks_with_context`] for details.
    pub fn chunk_indices_with_context<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = (usize, &'text str, HeadingContext<'text>)> + 'splitter {
        // One markdown parse for both element ranges (for chunking) and
        // heading metadata (for context).
        let (elements, headings) = parse_markdown(text);
        let mut hi = 0;
        let mut ctx = HeadingContext::default();
        TextChunks::<Sizer, Element>::new(&self.chunk_config, text, elements, Self::TRIM_CONST).map(
            move |(byte_offset, chunk)| {
                advance_context(&mut ctx, &headings, &mut hi, byte_offset);
                (byte_offset, chunk, ctx)
            },
        )
    }

    /// Like [`chunk_char_indices`] but additionally yields a
    /// [`HeadingContext`] reflecting the heading hierarchy active at the
    /// chunk's byte offset.
    ///
    /// See [`MarkdownSplitter::chunks_with_context`] for details.
    pub fn chunk_char_indices_with_context<'splitter, 'text: 'splitter>(
        &'splitter self,
        text: &'text str,
    ) -> impl Iterator<Item = ChunkCharIndexWithContext<'text>> + 'splitter {
        let (elements, headings) = parse_markdown(text);
        let mut hi = 0;
        let mut ctx = HeadingContext::default();
        TextChunksWithCharIndices::<Sizer, Element>::new(
            &self.chunk_config,
            text,
            elements,
            Self::TRIM_CONST,
        )
        .map(move |cc| {
            advance_context(&mut ctx, &headings, &mut hi, cc.byte_offset);
            ChunkCharIndexWithContext {
                chunk: cc.chunk,
                byte_offset: cc.byte_offset,
                char_offset: cc.char_offset,
                context: ctx,
            }
        })
    }

    /// Mirrors `<Self as Splitter<_>>::TRIM` for use in the inherent context
    /// methods (which can't easily reference the trait constant in expression
    /// position without UFCS gymnastics).
    const TRIM_CONST: Trim = Trim::PreserveIndentation;
}

impl<Sizer> Splitter<Sizer> for MarkdownSplitter<Sizer>
where
    Sizer: ChunkSizer,
{
    type Level = Element;

    const TRIM: Trim = Trim::PreserveIndentation;

    fn chunk_config(&self) -> &ChunkConfig<Sizer> {
        &self.chunk_config
    }

    fn parse(&self, text: &str) -> Vec<(Self::Level, Range<usize>)> {
        Parser::new_ext(text, Options::all())
            .into_offset_iter()
            .filter_map(|(event, range)| classify_event(&event).map(|el| (el, range)))
            .collect()
    }
}

/// Classify a pulldown-cmark event into the corresponding [`Element`] semantic
/// level, or `None` if the event doesn't open one. Shared by [`MarkdownSplitter`]'s
/// [`Splitter::parse`] impl and the combined [`parse_markdown`] walker so the
/// classification stays in one place.
fn classify_event(event: &Event<'_>) -> Option<Element> {
    match event {
        Event::Start(
            Tag::Emphasis
            | Tag::Strong
            | Tag::Strikethrough
            | Tag::Link { .. }
            | Tag::Image { .. }
            | Tag::Subscript
            | Tag::Superscript
            | Tag::TableCell,
        )
        | Event::Text(_)
        | Event::HardBreak
        | Event::Code(_)
        | Event::InlineHtml(_)
        | Event::InlineMath(_)
        | Event::FootnoteReference(_)
        | Event::TaskListMarker(_) => Some(Element::Inline),
        Event::SoftBreak => Some(Element::SoftBreak),
        Event::Html(_)
        | Event::DisplayMath(_)
        | Event::Start(
            Tag::Paragraph
            | Tag::CodeBlock(_)
            | Tag::FootnoteDefinition(_)
            | Tag::MetadataBlock(_)
            | Tag::TableHead
            | Tag::BlockQuote(_)
            | Tag::TableRow
            | Tag::Item
            | Tag::HtmlBlock
            | Tag::List(_)
            | Tag::Table(_)
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::DefinitionListDefinition,
        ) => Some(Element::Block),
        Event::Rule => Some(Element::Rule),
        Event::Start(Tag::Heading { level, .. }) => Some(Element::Heading((*level).into())),
        // End events are identical to start, so no need to grab them.
        Event::End(_) => None,
    }
}

/// Heading levels in markdown.
/// Sorted in reverse order for sorting purposes.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum HeadingLevel {
    /// Level 6 heading (`######`) — the smallest break.
    H6,
    /// Level 5 heading (`#####`).
    H5,
    /// Level 4 heading (`####`).
    H4,
    /// Level 3 heading (`###`).
    H3,
    /// Level 2 heading (`##`).
    H2,
    /// Level 1 heading (`#`) — the largest break.
    H1,
}

impl HeadingLevel {
    /// All six heading levels in shallow-to-deep order: `[H1, H2, …, H6]`.
    ///
    /// Useful for iterating over a [`HeadingContext`] without depending on
    /// the enum's declaration order (which is reversed for `Ord` purposes).
    pub const ALL: [Self; 6] = [Self::H1, Self::H2, Self::H3, Self::H4, Self::H5, Self::H6];

    /// The heading depth: `H1 → 1`, `H2 → 2`, … `H6 → 6`. Matches the number
    /// of `#` characters in the source (and the variant name).
    #[must_use]
    pub const fn depth(self) -> usize {
        match self {
            Self::H1 => 1,
            Self::H2 => 2,
            Self::H3 => 3,
            Self::H4 => 4,
            Self::H5 => 5,
            Self::H6 => 6,
        }
    }
}

impl From<pulldown_cmark::HeadingLevel> for HeadingLevel {
    fn from(value: pulldown_cmark::HeadingLevel) -> Self {
        match value {
            pulldown_cmark::HeadingLevel::H1 => HeadingLevel::H1,
            pulldown_cmark::HeadingLevel::H2 => HeadingLevel::H2,
            pulldown_cmark::HeadingLevel::H3 => HeadingLevel::H3,
            pulldown_cmark::HeadingLevel::H4 => HeadingLevel::H4,
            pulldown_cmark::HeadingLevel::H5 => HeadingLevel::H5,
            pulldown_cmark::HeadingLevel::H6 => HeadingLevel::H6,
        }
    }
}

/// How a particular semantic level relates to surrounding text elements.
#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum SemanticSplitPosition {
    /// The semantic level should be treated as its own chunk.
    Own,
    /// The semantic level should be included in the next chunk.
    Next,
}

/// Different semantic levels that text can be split by.
/// Each level provides a method of splitting text into chunks of a given level
/// as well as a fallback in case a given fallback is too large.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum Element {
    /// Single line break, which isn't necessarily a new element in Markdown
    SoftBreak,
    /// An inline element that is within a larger element such as a paragraph, but
    /// more specific than a sentence.
    Inline,
    /// Paragraph, code block, metadata, a row/item within a table or list, block quote, that can contain other "block" type elements, List or table that contains items
    Block,
    /// thematic break/horizontal rule
    Rule,
    /// Heading levels in markdown
    Heading(HeadingLevel),
}

impl Element {
    fn split_position(self) -> SemanticSplitPosition {
        match self {
            Self::SoftBreak | Self::Block | Self::Rule | Self::Inline => SemanticSplitPosition::Own,
            // Attach it to the next text
            Self::Heading(_) => SemanticSplitPosition::Next,
        }
    }

    fn treat_whitespace_as_previous(self) -> bool {
        match self {
            Self::SoftBreak | Self::Inline | Self::Rule | Self::Heading(_) => false,
            Self::Block => true,
        }
    }
}

impl SemanticLevel for Element {
    fn sections(
        text: &str,
        level_ranges: impl Iterator<Item = (Self, Range<usize>)>,
    ) -> impl Iterator<Item = (usize, &str)> {
        let mut cursor = 0;
        let mut final_match = false;
        level_ranges
            .batching(move |it| {
                loop {
                    match it.next() {
                        // If we've hit the end, actually return None
                        None if final_match => return None,
                        // First time we hit None, return the final section of the text
                        None => {
                            final_match = true;
                            return text.get(cursor..).map(|t| Either::Left(once((cursor, t))));
                        }
                        // Return text preceding match + the match
                        Some((level, range)) => {
                            let offset = cursor;
                            match level.split_position() {
                                SemanticSplitPosition::Own => {
                                    if range.start < cursor {
                                        continue;
                                    }
                                    let prev_section = text
                                        .get(cursor..range.start)
                                        .expect("invalid character sequence");
                                    if level.treat_whitespace_as_previous()
                                        && prev_section.chars().all(char::is_whitespace)
                                    {
                                        let section = text
                                            .get(cursor..range.end)
                                            .expect("invalid character sequence");
                                        cursor = range.end;
                                        return Some(Either::Left(once((offset, section))));
                                    }
                                    let separator = text
                                        .get(range.start..range.end)
                                        .expect("invalid character sequence");
                                    cursor = range.end;
                                    return Some(Either::Right(
                                        [(offset, prev_section), (range.start, separator)]
                                            .into_iter(),
                                    ));
                                }
                                SemanticSplitPosition::Next => {
                                    if range.start < cursor {
                                        continue;
                                    }
                                    let prev_section = text
                                        .get(cursor..range.start)
                                        .expect("invalid character sequence");
                                    // Separator will be part of the next chunk
                                    cursor = range.start;
                                    return Some(Either::Left(once((offset, prev_section))));
                                }
                            }
                        }
                    }
                }
            })
            .flatten()
            .filter(|(_, s)| !s.is_empty())
    }
}

/// The heading hierarchy active at a given point in the document.
///
/// Returned alongside each chunk by [`MarkdownSplitter::chunks_with_context`]
/// and friends. Internally a fixed-size array, so cloning / iterating is
/// allocation-free.
///
/// Semantics: when a heading at level `L` appears in the source, it replaces
/// the previous entry at level `L` and clears every entry at deeper levels
/// (consistent with how markdown headings nest).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeadingContext<'text> {
    levels: [Option<&'text str>; 6],
}

impl<'text> HeadingContext<'text> {
    /// Heading text at the given level, if one is active.
    #[must_use]
    pub fn at(&self, level: HeadingLevel) -> Option<&'text str> {
        self.levels[level.depth() - 1]
    }

    /// The deepest active heading: `(level, text)`. Returns `None` if no
    /// heading has been seen yet (preamble content).
    #[must_use]
    pub fn deepest(&self) -> Option<(HeadingLevel, &'text str)> {
        HeadingLevel::ALL
            .iter()
            .rev()
            .find_map(|&l| self.levels[l.depth() - 1].map(|t| (l, t)))
    }

    /// Iterate active headings from H1 → H6 (shallow to deep).
    pub fn iter(&self) -> impl Iterator<Item = (HeadingLevel, &'text str)> + '_ {
        HeadingLevel::ALL
            .iter()
            .filter_map(move |&l| self.levels[l.depth() - 1].map(|t| (l, t)))
    }

    fn set(&mut self, level: HeadingLevel, text: &'text str) {
        let idx = level.depth() - 1;
        self.levels[idx] = Some(text);
        // Clear deeper levels — a new H2 invalidates the previous H3/H4/…
        for deeper in &mut self.levels[(idx + 1)..] {
            *deeper = None;
        }
    }
}

/// A chunk plus its byte/char offsets *and* the heading hierarchy active at
/// its byte offset. Returned by
/// [`MarkdownSplitter::chunk_char_indices_with_context`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkCharIndexWithContext<'text> {
    /// The text of the generated chunk.
    pub chunk: &'text str,
    /// The byte offset of the chunk within the original text.
    pub byte_offset: usize,
    /// The character offset of the chunk within the original text.
    pub char_offset: usize,
    /// Heading hierarchy active at this chunk's byte offset.
    pub context: HeadingContext<'text>,
}

/// `(start_byte, level, text_slice)` per heading.
type HeadingRecord<'text> = (usize, HeadingLevel, &'text str);

/// In-flight heading state during the parse walk:
/// `(start_byte, level, Option<(inner_first_byte, inner_last_byte)>)`.
type CurrentHeading = (usize, HeadingLevel, Option<(usize, usize)>);

/// Walk the markdown source via pulldown-cmark **once**, producing both the
/// element ranges needed to chunk and the heading metadata needed for context.
///
/// Element classification matches [`classify_event`] / [`Splitter::parse`];
/// heading text is collected as a sub-slice of `text` spanning from the first
/// Text/Code event inside the heading to the last (pulldown-cmark already
/// excludes the `#` markers / setext underline). Headings nested inside block
/// quotes, list items, or footnote definitions are skipped — they don't
/// define document structure.
fn parse_markdown(text: &str) -> (Vec<(Element, Range<usize>)>, Vec<HeadingRecord<'_>>) {
    let mut elements: Vec<(Element, Range<usize>)> = Vec::new();
    let mut headings: Vec<HeadingRecord<'_>> = Vec::new();
    let mut container_depth: usize = 0;
    let mut current_heading: Option<CurrentHeading> = None;

    for (event, range) in Parser::new_ext(text, Options::all()).into_offset_iter() {
        if let Some(el) = classify_event(&event) {
            elements.push((el, range.clone()));
        }
        match &event {
            Event::Start(Tag::BlockQuote(_) | Tag::Item | Tag::FootnoteDefinition(_)) => {
                container_depth += 1;
            }
            Event::End(TagEnd::BlockQuote(_) | TagEnd::Item | TagEnd::FootnoteDefinition) => {
                container_depth = container_depth.saturating_sub(1);
            }
            Event::Start(Tag::Heading { level, .. }) if container_depth == 0 => {
                current_heading = Some((range.start, (*level).into(), None));
            }
            Event::End(TagEnd::Heading(_)) if current_heading.is_some() => {
                if let Some((start, level, inner_range)) = current_heading.take() {
                    let inner = match inner_range {
                        Some((s, e)) => &text[s..e],
                        None => "",
                    };
                    headings.push((start, level, inner));
                }
            }
            Event::Text(_) | Event::Code(_) if current_heading.is_some() => {
                if let Some((_, _, ref mut inner_range)) = current_heading {
                    match inner_range {
                        Some((_, end)) => *end = range.end,
                        None => *inner_range = Some((range.start, range.end)),
                    }
                }
            }
            _ => {}
        }
    }
    (elements, headings)
}

/// Advance `ctx` so it reflects every heading whose byte offset is `<=
/// byte_offset`. Mutates `*idx` so subsequent calls resume where we left off.
fn advance_context<'text>(
    ctx: &mut HeadingContext<'text>,
    headings: &[HeadingRecord<'text>],
    idx: &mut usize,
    byte_offset: usize,
) {
    while *idx < headings.len() && headings[*idx].0 <= byte_offset {
        let (_, level, text) = headings[*idx];
        ctx.set(level, text);
        *idx += 1;
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::min;

    use fake::{Fake, Faker};

    use crate::splitter::SemanticSplitRanges;

    use super::*;

    #[test]
    fn returns_one_chunk_if_text_is_shorter_than_max_chunk_size() {
        let text = Faker.fake::<String>();
        let chunks = MarkdownSplitter::new(ChunkConfig::new(text.chars().count()).with_trim(false))
            .chunks(&text)
            .collect::<Vec<_>>();

        assert_eq!(vec![&text], chunks);
    }

    #[test]
    fn returns_two_chunks_if_text_is_longer_than_max_chunk_size() {
        let text1 = Faker.fake::<String>();
        let text2 = Faker.fake::<String>();
        let text = format!("{text1}{text2}");
        // Round up to one above half so it goes to 2 chunks
        let max_chunk_size = text.chars().count() / 2 + 1;

        let chunks = MarkdownSplitter::new(ChunkConfig::new(max_chunk_size).with_trim(false))
            .chunks(&text)
            .collect::<Vec<_>>();

        assert!(chunks.iter().all(|c| c.chars().count() <= max_chunk_size));

        // Check that beginning of first chunk and text 1 matches
        let len = min(text1.len(), chunks[0].len());
        assert_eq!(text1[..len], chunks[0][..len]);
        // Check that end of second chunk and text 2 matches
        let len = min(text2.len(), chunks[1].len());
        assert_eq!(
            text2[(text2.len() - len)..],
            chunks[1][chunks[1].len() - len..]
        );

        assert_eq!(chunks.join(""), text);
    }

    #[test]
    fn empty_string() {
        let text = "";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(100).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert!(chunks.is_empty());
    }

    #[test]
    fn can_handle_unicode_characters() {
        let text = "éé"; // Char that is more than one byte
        let chunks = MarkdownSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["é", "é"], chunks);
    }

    #[test]
    fn chunk_by_graphemes() {
        let text = "a̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(3).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        // \r\n is grouped together not separated
        assert_eq!(vec!["a̐é", "ö̲", "\r\n"], chunks);
    }

    #[test]
    fn trim_char_indices() {
        let text = " a b ";
        let chunks = MarkdownSplitter::new(1)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(vec![(1, "a"), (3, "b")], chunks);
    }

    #[test]
    fn chunk_char_indices() {
        let text = " a b ";
        let chunks = MarkdownSplitter::new(1)
            .chunk_char_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                ChunkCharIndex {
                    chunk: "a",
                    byte_offset: 1,
                    char_offset: 1
                },
                ChunkCharIndex {
                    chunk: "b",
                    byte_offset: 3,
                    char_offset: 3,
                },
            ],
            chunks
        );
    }

    #[test]
    fn graphemes_fallback_to_chars() {
        let text = "a̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(1).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec!["a", "\u{310}", "é", "ö", "\u{332}", "\r", "\n"],
            chunks
        );
    }

    #[test]
    fn trim_grapheme_indices() {
        let text = "\r\na̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(3)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(vec![(2, "a̐é"), (7, "ö̲")], chunks);
    }

    #[test]
    fn grapheme_char_indices() {
        let text = "\r\na̐éö̲\r\n";
        let chunks = MarkdownSplitter::new(3)
            .chunk_char_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                ChunkCharIndex {
                    chunk: "a̐é",
                    byte_offset: 2,
                    char_offset: 2
                },
                ChunkCharIndex {
                    chunk: "ö̲",
                    byte_offset: 7,
                    char_offset: 5
                }
            ],
            chunks
        );
    }

    #[test]
    fn chunk_by_words() {
        let text = "The quick brown fox can jump 32.3 feet, right?";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(10).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![
                "The quick ",
                "brown fox ",
                "can jump ",
                "32.3 feet,",
                " right?"
            ],
            chunks
        );
    }

    #[test]
    fn words_fallback_to_graphemes() {
        let text = "Thé quick\r\n";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(2).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["Th", "é ", "qu", "ic", "k", "\r\n"], chunks);
    }

    #[test]
    fn trim_word_indices() {
        let text = "Some text from a document";
        let chunks = MarkdownSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![(0, "Some text"), (10, "from a"), (17, "document")],
            chunks
        );
    }

    #[test]
    fn chunk_by_sentences() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(21).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(vec!["Mr. Fox jumped. ", "The dog was too lazy."], chunks);
    }

    #[test]
    fn sentences_falls_back_to_words() {
        let text = "Mr. Fox jumped. The dog was too lazy.";
        let chunks = MarkdownSplitter::new(ChunkConfig::new(16).with_trim(false))
            .chunks(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec!["Mr. Fox jumped. ", "The dog was too ", "lazy."],
            chunks
        );
    }

    #[test]
    fn trim_sentence_indices() {
        let text = "Some text. From a document.";
        let chunks = MarkdownSplitter::new(10)
            .chunk_indices(text)
            .collect::<Vec<_>>();

        assert_eq!(
            vec![(0, "Some text."), (11, "From a"), (18, "document.")],
            chunks
        );
    }

    #[test]
    fn test_no_markdown_separators() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("Some text without any markdown separators"));

        assert_eq!(
            vec![(Element::Block, 0..41), (Element::Inline, 0..41)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_checklist() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("- [ ] incomplete task\n- [x] completed task"));

        assert_eq!(
            vec![
                (Element::Block, 0..42),
                (Element::Block, 0..22),
                (Element::Inline, 2..5),
                (Element::Inline, 6..21),
                (Element::Block, 22..42),
                (Element::Inline, 24..27),
                (Element::Inline, 28..42),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_reference() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Footnote[^1]"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..8),
                (Element::Inline, 8..12),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_code() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("`bash`"));

        assert_eq!(
            vec![(Element::Block, 0..6), (Element::Inline, 0..6)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_emphasis() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("*emphasis*"));

        assert_eq!(
            vec![
                (Element::Block, 0..10),
                (Element::Inline, 0..10),
                (Element::Inline, 1..9),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strong() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("**emphasis**"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_strikethrough() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("~~emphasis~~"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..10),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_link() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("[link](url)"));

        assert_eq!(
            vec![
                (Element::Block, 0..11),
                (Element::Inline, 0..11),
                (Element::Inline, 1..5),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_image() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("![link](url)"));

        assert_eq!(
            vec![
                (Element::Block, 0..12),
                (Element::Inline, 0..12),
                (Element::Inline, 2..6),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_inline_html() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("<span>Some text</span>"));

        assert_eq!(
            vec![
                (Element::Block, 0..22),
                (Element::Inline, 0..6),
                (Element::Inline, 6..15),
                (Element::Inline, 15..22),
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_html() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("<div>Some text</div>"));

        assert_eq!(
            vec![(Element::Block, 0..20), (Element::Block, 0..20)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_table() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(
            splitter.parse("| Header 1 | Header 2 |\n| --- | --- |\n| Cell 1 | Cell 2 |"),
        );
        assert_eq!(
            vec![
                (Element::Block, 0..57),
                (Element::Block, 0..24),
                (Element::Inline, 1..11),
                (Element::Inline, 2..10),
                (Element::Inline, 12..22),
                (Element::Inline, 13..21),
                (Element::Block, 38..57),
                (Element::Inline, 39..47),
                (Element::Inline, 40..46),
                (Element::Inline, 48..56),
                (Element::Inline, 49..55)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_softbreak() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\nwith a softbreak"));

        assert_eq!(
            vec![
                (Element::Block, 0..26),
                (Element::Inline, 0..9),
                (Element::SoftBreak, 9..10),
                (Element::Inline, 10..26)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_hardbreak() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\\\nwith a hardbreak"));

        assert_eq!(
            vec![
                (Element::Block, 0..27),
                (Element::Inline, 0..9),
                (Element::Inline, 9..11),
                (Element::Inline, 11..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_footnote_def() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("[^first]: Footnote"));

        assert_eq!(
            vec![
                (Element::Block, 0..18),
                (Element::Block, 10..18),
                (Element::Inline, 10..18)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_code_block() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("```\ncode\n```"));

        assert_eq!(
            vec![(Element::Block, 0..12), (Element::Inline, 4..9)],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_block_quote() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("> quote"));

        assert_eq!(
            vec![
                (Element::Block, 0..7),
                (Element::Block, 2..7),
                (Element::Inline, 2..7)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_with_rule() {
        let splitter = MarkdownSplitter::new(10);
        let markdown = SemanticSplitRanges::new(splitter.parse("Some text\n\n---\n\nwith a rule"));

        assert_eq!(
            vec![
                (Element::Block, 0..10),
                (Element::Inline, 0..9),
                (Element::Rule, 11..15),
                (Element::Block, 16..27),
                (Element::Inline, 16..27)
            ],
            markdown.ranges_after_offset(0).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_heading() {
        for (index, (heading, level)) in [
            ("#", HeadingLevel::H1),
            ("##", HeadingLevel::H2),
            ("###", HeadingLevel::H3),
            ("####", HeadingLevel::H4),
            ("#####", HeadingLevel::H5),
            ("######", HeadingLevel::H6),
        ]
        .into_iter()
        .enumerate()
        {
            let splitter = MarkdownSplitter::new(10);
            let markdown = SemanticSplitRanges::new(splitter.parse(&format!("{heading} Heading")));

            assert_eq!(
                vec![
                    (Element::Heading(level), 0..9 + index),
                    (Element::Inline, 2 + index..9 + index)
                ],
                markdown.ranges_after_offset(0).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_ranges_after_offset_block() {
        let splitter = MarkdownSplitter::new(10);
        let markdown =
            SemanticSplitRanges::new(splitter.parse("- [ ] incomplete task\n- [x] completed task"));

        assert_eq!(
            vec![(Element::Block, 0..22), (Element::Block, 22..42),],
            markdown
                .level_ranges_after_offset(0, Element::Block)
                .collect::<Vec<_>>()
        );
    }

    // ------------------------------------------------------------------
    // chunks_with_context / HeadingContext
    // ------------------------------------------------------------------

    #[test]
    fn context_empty_for_doc_without_headings() {
        let splitter = MarkdownSplitter::new(64);
        let chunks: Vec<_> = splitter
            .chunks_with_context("hello world\n\nsecond paragraph")
            .collect();
        assert!(!chunks.is_empty());
        for (_, ctx) in &chunks {
            assert_eq!(ctx.deepest(), None);
            assert!(ctx.iter().next().is_none());
        }
    }

    #[test]
    fn context_tracks_atx_heading_levels() {
        // Small capacity forces text-splitter to emit one chunk per section,
        // so each chunk's byte_offset advances the context through nested headings.
        let splitter = MarkdownSplitter::new(20);
        let md = "# Top\n\nintro\n\n## Sub\n\nbody\n\n### Deep\n\ndeepbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();

        // The last chunk sits past all three headings.
        let (_, last_ctx) = chunks.last().expect("at least one chunk");
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some("Top"));
        assert_eq!(last_ctx.at(HeadingLevel::H2), Some("Sub"));
        assert_eq!(last_ctx.at(HeadingLevel::H3), Some("Deep"));
        assert_eq!(last_ctx.deepest(), Some((HeadingLevel::H3, "Deep")));
    }

    #[test]
    fn context_clears_deeper_levels_on_sibling_heading() {
        let splitter = MarkdownSplitter::new(20);
        let md = "# Top\n\nintro\n\n## A\n\n### Deep\n\nfoo\n\n## B\n\nbar\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();

        // Last chunk is under "## B"; H3 from the earlier "### Deep" must be cleared.
        let (_, last_ctx) = chunks.last().expect("at least one chunk");
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some("Top"));
        assert_eq!(last_ctx.at(HeadingLevel::H2), Some("B"));
        assert_eq!(last_ctx.at(HeadingLevel::H3), None);
    }

    #[test]
    fn context_recognises_setext_heading() {
        let splitter = MarkdownSplitter::new(64);
        let md = "Title\n=====\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let body_chunk = chunks
            .iter()
            .find(|(c, _)| c.contains("body"))
            .expect("body chunk");
        assert_eq!(body_chunk.1.at(HeadingLevel::H1), Some("Title"));
    }

    #[test]
    fn context_ignores_hash_inside_code_block() {
        let splitter = MarkdownSplitter::new(128);
        let md = "# Real\n\n```bash\n# shell comment, not a heading\n```\n\nafter\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        for (_, ctx) in &chunks {
            // The only heading we should ever see is "Real".
            for (_, text) in ctx.iter() {
                assert_eq!(text, "Real");
            }
        }
    }

    #[test]
    fn context_strips_atx_trailing_hashes() {
        let splitter = MarkdownSplitter::new(64);
        let md = "## Hello ##\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let body_chunk = chunks
            .iter()
            .find(|(c, _)| c.contains("body"))
            .expect("body chunk");
        assert_eq!(body_chunk.1.at(HeadingLevel::H2), Some("Hello"));
    }

    #[test]
    fn chunk_indices_with_context_returns_byte_offsets() {
        let splitter = MarkdownSplitter::new(64);
        let md = "# A\n\nfoo\n\n# B\n\nbar\n";
        let collected: Vec<_> = splitter.chunk_indices_with_context(md).collect();
        // Every entry has (byte_offset, chunk_text, context).
        for (byte_offset, chunk, _) in &collected {
            assert_eq!(&md[*byte_offset..*byte_offset + chunk.len()], *chunk);
        }
    }

    #[test]
    fn chunk_char_indices_with_context_returns_full_struct() {
        let splitter = MarkdownSplitter::new(64);
        let md = "# A\n\nfoo\n\n# B\n\nbar\n";
        let collected: Vec<_> = splitter.chunk_char_indices_with_context(md).collect();
        assert!(!collected.is_empty());
        for c in &collected {
            assert_eq!(&md[c.byte_offset..c.byte_offset + c.chunk.len()], c.chunk);
            // char_offset never exceeds byte_offset (chars ≤ bytes in UTF-8).
            assert!(c.char_offset <= c.byte_offset);
        }
    }

    #[test]
    fn context_iter_yields_h1_to_h6_order() {
        // Sanity check that HeadingContext::iter walks shallowest → deepest.
        let mut ctx = HeadingContext::default();
        ctx.set(HeadingLevel::H3, "deep");
        ctx.set(HeadingLevel::H1, "top");
        // Setting H1 clears deeper levels per the docs.
        let levels: Vec<_> = ctx.iter().collect();
        assert_eq!(levels, vec![(HeadingLevel::H1, "top")]);
    }

    // ------------------------------------------------------------------
    // Weird heading cases — pin behavior against pathological markdown.
    // ------------------------------------------------------------------

    #[test]
    fn context_handles_skipped_heading_levels() {
        // `# A` directly to `### C` with no `##` in between. H2 stays vacant.
        let splitter = MarkdownSplitter::new(20);
        let md = "# A\n\nintro\n\n### C\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let (_, last_ctx) = chunks.last().unwrap();
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some("A"));
        assert_eq!(last_ctx.at(HeadingLevel::H2), None);
        assert_eq!(last_ctx.at(HeadingLevel::H3), Some("C"));
        assert_eq!(last_ctx.deepest(), Some((HeadingLevel::H3, "C")));
    }

    #[test]
    fn context_handles_multiple_h1s() {
        let splitter = MarkdownSplitter::new(15);
        let md = "# First\n\nfoo\n\n# Second\n\nbar\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let (_, last_ctx) = chunks.last().unwrap();
        // Second H1 replaces the first; deeper levels (none here) cleared.
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some("Second"));
        assert_eq!(last_ctx.deepest(), Some((HeadingLevel::H1, "Second")));
    }

    #[test]
    fn context_handles_empty_heading_text() {
        // CommonMark: `#` alone is a level-1 heading with empty content.
        let splitter = MarkdownSplitter::new(20);
        let md = "#\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let (_, last_ctx) = chunks.last().unwrap();
        // We still register the level, just with empty text.
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some(""));
    }

    #[test]
    fn context_handles_inline_formatting_in_heading() {
        // pulldown-cmark emits Text events for the words but skips the
        // `**` / `*` markers. Our text slice spans from the first to last
        // Text event, so any markers between events are kept; markers
        // strictly outside the first/last event are dropped.
        let splitter = MarkdownSplitter::new(30);
        let md = "## **bold** middle *em*\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let (_, last_ctx) = chunks.last().unwrap();
        let h2 = last_ctx.at(HeadingLevel::H2).unwrap();
        // Both content words must appear; exact marker representation is
        // implementation detail of pulldown-cmark.
        assert!(h2.contains("bold"), "expected `bold` in {h2:?}");
        assert!(h2.contains("middle"), "expected `middle` in {h2:?}");
        assert!(h2.contains("em"), "expected `em` in {h2:?}");
    }

    #[test]
    fn context_ignores_heading_inside_blockquote() {
        // Capacity small enough to force the splitter past byte_offset = 10
        // where pulldown-cmark *does* emit Tag::Heading for `> ## Quoted`.
        // Without the container-depth filter, the chunk past the blockquote
        // would see ctx[H2] == Some("Quoted").
        let splitter = MarkdownSplitter::new(15);
        let md = "# Real\n\n> ## Quoted\n>\n> body of quote\n\nafter\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        // No chunk should see "Quoted" as H2 — it lives inside a block quote.
        for (_, ctx) in &chunks {
            assert_ne!(
                ctx.at(HeadingLevel::H2),
                Some("Quoted"),
                "blockquoted heading leaked into context"
            );
        }
        // The legitimate H1 is still picked up by some chunk.
        let any_with_real = chunks
            .iter()
            .any(|(_, c)| c.at(HeadingLevel::H1) == Some("Real"));
        assert!(any_with_real);
    }

    #[test]
    fn context_ignores_heading_inside_list_item() {
        let splitter = MarkdownSplitter::new(15);
        let md = "# Real\n\n- ## Listed\n  body\n\nafter\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        for (_, ctx) in &chunks {
            assert_ne!(
                ctx.at(HeadingLevel::H2),
                Some("Listed"),
                "list-item heading leaked into context"
            );
        }
    }

    #[test]
    fn context_handles_setext_alongside_atx() {
        // Setext H1 followed by ATX H2 should produce a normal H1 → H2 chain.
        let splitter = MarkdownSplitter::new(25);
        let md = "Title\n=====\n\nintro\n\n## Sub\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        let (_, last_ctx) = chunks.last().unwrap();
        assert_eq!(last_ctx.at(HeadingLevel::H1), Some("Title"));
        assert_eq!(last_ctx.at(HeadingLevel::H2), Some("Sub"));
    }

    #[test]
    fn context_handles_html_pseudo_heading_as_non_heading() {
        // `<h1>...</h1>` is HTML, not a markdown heading — pulldown emits
        // Event::Html, not Tag::Heading, so we ignore it.
        let splitter = MarkdownSplitter::new(64);
        let md = "<h1>HTML pretender</h1>\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        for (_, ctx) in &chunks {
            assert_eq!(
                ctx.deepest(),
                None,
                "no markdown heading should be registered"
            );
        }
    }

    #[test]
    fn context_handles_more_than_six_hashes_as_paragraph() {
        // 7 `#`s isn't a valid ATX heading; it's a paragraph.
        let splitter = MarkdownSplitter::new(64);
        let md = "####### not a heading\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        for (_, ctx) in &chunks {
            assert_eq!(ctx.deepest(), None);
        }
    }

    #[test]
    fn context_handles_hash_without_space() {
        // `#foo` (no space after #) is not a heading per CommonMark.
        let splitter = MarkdownSplitter::new(64);
        let md = "#foo\n\nbody\n";
        let chunks: Vec<_> = splitter.chunks_with_context(md).collect();
        for (_, ctx) in &chunks {
            assert_eq!(ctx.deepest(), None);
        }
    }

    #[test]
    fn context_set_clears_deeper_levels() {
        let mut ctx = HeadingContext::default();
        ctx.set(HeadingLevel::H1, "A");
        ctx.set(HeadingLevel::H2, "B");
        ctx.set(HeadingLevel::H3, "C");
        assert_eq!(ctx.deepest(), Some((HeadingLevel::H3, "C")));
        // Setting a new H2 should clear H3.
        ctx.set(HeadingLevel::H2, "B2");
        assert_eq!(ctx.at(HeadingLevel::H2), Some("B2"));
        assert_eq!(ctx.at(HeadingLevel::H3), None);
        assert_eq!(ctx.deepest(), Some((HeadingLevel::H2, "B2")));
    }
}
