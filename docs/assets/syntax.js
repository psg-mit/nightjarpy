// Simple Python syntax highlighter with custom color scheme
document.addEventListener('DOMContentLoaded', function() {
  const codeBlocks = document.querySelectorAll('pre code');
  
  codeBlocks.forEach(block => {
    // Get the text content (this will decode HTML entities automatically)
    const code = block.textContent;
    const highlighted = highlightPython(code);
    block.innerHTML = highlighted;
  });
});

function escapeHtml(text) {
  const map = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;'
  };
  return text.replace(/[&<>"']/g, m => map[m]);
}

function highlightPython(code) {
  // Python keywords
  const keywords = /\b(import|from|as|def|class|return|if|elif|else|for|while|in|is|not|and|or|True|False|None|with|try|except|finally|raise|break|continue|pass|lambda|yield|async|await)\b/g;
  
  // Decorators
  const decorators = /(@[a-zA-Z_][a-zA-Z0-9_.]*)/g;
  
  // Strings (handle both single and double quotes, and triple quotes)
  const strings = /("""[\s\S]*?"""|'''[\s\S]*?'''|f"(?:[^"\\]|\\.)*?"|"(?:[^"\\]|\\.)*?"|'(?:[^'\\]|\\.)*?')/g;
  
  // Comments
  const comments = /(#.*$)/gm;
  
  // Numbers
  const numbers = /\b(\d+\.?\d*)\b/g;
  
  // Built-in functions
  const builtins = /\b(print|len|range|list|dict|str|int|float|bool|type|isinstance|enumerate|zip|map|filter|open|input)\b/g;
  
  // First, protect strings and comments from being processed
  const protectedStrings = [];
  const protectedComments = [];
  
  let result = code;
  
  result = result.replace(strings, (match) => {
    protectedStrings.push(match);
    return `__STRING_${protectedStrings.length - 1}__`;
  });
  
  result = result.replace(comments, (match) => {
    protectedComments.push(match);
    return `__COMMENT_${protectedComments.length - 1}__`;
  });
  
  // Apply syntax highlighting on unescaped text
  result = result.replace(decorators, '«decorator»$1«/decorator»');
  result = result.replace(/\b(def)\s+([a-zA-Z_][a-zA-Z0-9_]*)/g, '«keyword»$1«/keyword» «function»$2«/function»');
  result = result.replace(keywords, '«keyword»$1«/keyword»');
  result = result.replace(builtins, '«builtin»$1«/builtin»');
  result = result.replace(numbers, '«number»$1«/number»');
  
  // Now escape all HTML
  result = escapeHtml(result);
  
  // Convert our markers back to proper HTML spans
  result = result.replace(/«(\w+)»/g, '<span class="token $1">');
  result = result.replace(/«\/(\w+)»/g, '</span>');
  
  // Restore comments with proper escaping
  protectedComments.forEach((comment, i) => {
    result = result.replace(`__COMMENT_${i}__`, `<span class="token comment">${escapeHtml(comment)}</span>`);
  });
  
  // Restore strings with proper escaping, formatting natural blocks
  protectedStrings.forEach((str, i) => {
    const replacement = formatStringToken(str);
    result = result.replace(`__STRING_${i}__`, replacement);
  });
  
  return result;
}

function formatStringToken(str) {
  const trimmed = str.trimStart();
  if (trimmed.startsWith('"""natural') || trimmed.startsWith("'''natural")) {
    return wrapNaturalString(str);
  }
  return `<span class="token string">${escapeHtml(str)}</span>`;
}

function wrapNaturalString(str) {
  const leadingMatch = str.match(/^\s*/);
  const leading = leadingMatch ? leadingMatch[0] : '';
  const withoutLeading = str.slice(leading.length);
  const quote = withoutLeading.startsWith("'''") ? "'''" : '"""';
  const newlineIndex = withoutLeading.indexOf('\n');
  if (newlineIndex === -1) {
    return `<span class="token string">${escapeHtml(str)}</span>`;
  }
  const header = withoutLeading.slice(0, newlineIndex + 1);
  const remainder = withoutLeading.slice(newlineIndex + 1);
  const closingIndex = remainder.lastIndexOf(quote);
  if (closingIndex === -1) {
    return `<span class="token string">${escapeHtml(str)}</span>`;
  }
  const beforeClosing = remainder.slice(0, closingIndex);
  const trailingWhitespaceMatch = beforeClosing.match(/(\s*)$/);
  const trailingWhitespace = trailingWhitespaceMatch ? trailingWhitespaceMatch[0] : '';
  const bodyRaw = trailingWhitespace ? beforeClosing.slice(0, -trailingWhitespace.length) : beforeClosing;
  const newlineIndentMatch = trailingWhitespace.match(/((?:\r?\n)+)([ \t]*)$/);
  const newlineSegment = newlineIndentMatch ? newlineIndentMatch[1] : '';
  const closingIndent = newlineIndentMatch ? newlineIndentMatch[2] : '';
  const footer = closingIndent + remainder.slice(closingIndex);
  const { normalizedBody, indentWidth } = normalizeNaturalBody(bodyRaw);
  const indentValue = indentWidth > 0 ? `${indentWidth}ch` : '0ch';
  const headerHtml = escapeHtml(leading + header);
  const bodyContent = normalizedBody + newlineSegment;
  const bodyHtml = `<span class="natural-block" style="--natural-indent:${indentValue};">${escapeHtml(bodyContent)}</span>`;
  const footerHtml = escapeHtml(footer);
  return `<span class="token string">${headerHtml}${bodyHtml}${footerHtml}</span>`;
}

function normalizeNaturalBody(body) {
  const lines = body.split(/\r?\n/);
  let minIndent = null;
  lines.forEach(line => {
    if (!line.trim()) return;
    const match = line.match(/^\s*/);
    const indent = match ? match[0].length : 0;
    if (minIndent === null || indent < minIndent) {
      minIndent = indent;
    }
  });
  const indentWidth = minIndent || 0;
  const normalizedBody = lines
    .map(line => line.slice(Math.min(indentWidth, line.length)))
    .join('\n');
  return { normalizedBody, indentWidth };
}
