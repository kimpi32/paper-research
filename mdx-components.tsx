import type { MDXComponents } from "mdx/types";
import { KeyIdea } from "@/components/content/KeyIdea";
import { Formula } from "@/components/content/Formula";
import { Impact } from "@/components/content/Impact";
import { FoundationFieldLinks } from "@/components/content/FoundationFieldLinks";

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    wrapper: ({ children }) => (
      <article className="mdx-content">{children}</article>
    ),
    h1: ({ children }) => (
      <h1 className="text-3xl font-bold mt-10 mb-5 text-gray-900 border-b border-gray-200 pb-3">
        {children}
      </h1>
    ),
    h2: ({ children }) => (
      <h2 className="text-xl font-semibold mt-10 mb-4 text-gray-800 border-b border-gray-100 pb-2">
        {children}
      </h2>
    ),
    h3: ({ children }) => (
      <h3 className="text-lg font-semibold mt-8 mb-3 text-gray-700">
        {children}
      </h3>
    ),
    p: ({ children }) => (
      <p className="text-base leading-7 mb-4 text-gray-700">{children}</p>
    ),
    ul: ({ children }) => (
      <ul className="list-disc pl-6 mb-5 text-gray-700 space-y-1.5 text-base leading-7">
        {children}
      </ul>
    ),
    ol: ({ children }) => (
      <ol className="list-decimal pl-6 mb-5 text-gray-700 space-y-1.5 text-base leading-7">
        {children}
      </ol>
    ),
    li: ({ children }) => <li className="pl-1">{children}</li>,
    a: ({ href, children }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 hover:text-blue-800 underline underline-offset-2"
      >
        {children}
      </a>
    ),
    strong: ({ children }) => (
      <strong className="font-semibold text-gray-900">{children}</strong>
    ),
    em: ({ children }) => <em className="italic">{children}</em>,
    hr: () => <hr className="my-10 border-gray-200" />,
    table: ({ children }) => (
      <div className="overflow-x-auto mb-6">
        <table className="min-w-full text-sm border border-gray-200 rounded-lg overflow-hidden">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }) => (
      <thead className="bg-gray-50 text-gray-600 text-left">{children}</thead>
    ),
    th: ({ children }) => (
      <th className="px-4 py-2.5 font-medium border-b border-gray-200">{children}</th>
    ),
    td: ({ children }) => (
      <td className="px-4 py-2 border-b border-gray-100 text-gray-700">{children}</td>
    ),
    blockquote: ({ children }) => (
      <blockquote className="border-l-4 border-gray-300 pl-4 my-4 text-gray-500 italic">
        {children}
      </blockquote>
    ),
    code: ({ children }) => (
      <code className="bg-gray-100 text-gray-800 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    ),
    KeyIdea,
    Formula,
    Impact,
    FoundationFieldLinks,
    ...components,
  };
}
