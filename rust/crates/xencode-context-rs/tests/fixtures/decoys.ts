// class CommentClass {}
// function commentFn() {}
/* interface BlockInterface {} */
/* enum BlockEnum {} */
const quotedClass = "class QuotedClass {}";
const quotedFn = 'function quotedFn() {}';
const template = `
class TemplateClass {}
function templateFn() {}
export const templateArrow = () => 1;
/** doc in a template that must not be read */
/* interface TemplateInterface {} */
`;
const pattern = /class\s+RegexClass|function\s+regexFn/g;
/** A real declaration. */
export function realFn(): void {}
// import fake from "fake-module";
/* export { fakeExport }; */
const divider = 10 / 2;
