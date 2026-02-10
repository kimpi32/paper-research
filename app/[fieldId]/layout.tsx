import { FieldSidebar } from "@/components/layout/Sidebar";
import { fields } from "@/lib/fields";

export function generateStaticParams() {
  return fields.map((field) => ({ fieldId: field.id }));
}

export default async function FieldLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ fieldId: string }>;
}) {
  const { fieldId } = await params;

  return (
    <div className="flex min-h-[calc(100vh-4rem)]">
      <FieldSidebar fieldId={fieldId} />
      <main className="flex-1">{children}</main>
    </div>
  );
}
