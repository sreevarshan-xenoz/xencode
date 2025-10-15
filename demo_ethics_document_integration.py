#!/usr/bin/env python3
"""
Demo: Ethics Framework Integration with Document Processing

This demo shows how the ethics framework integrates with document processing
to provide comprehensive content analysis and compliance checking.
"""

import asyncio
import tempfile
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import the integrated components
from xencode.document_processor import document_processor
from xencode.ethics_document_integration import document_ethics_processor


async def create_test_documents():
    """Create test documents with various content types"""
    test_docs = []
    
    # Document 1: Clean, professional content
    clean_content = """
    Welcome to our inclusive workplace policy document.
    
    Our company values diversity and treats all employees with respect and dignity.
    We believe that everyone deserves equal opportunities regardless of their background.
    
    Key principles:
    - Respect for all individuals
    - Equal opportunity employment
    - Professional communication standards
    - Collaborative work environment
    
    For questions, contact our HR department.
    """
    
    # Document 2: Content with bias issues
    biased_content = """
    Technical Team Requirements
    
    We are looking for developers for our engineering team.
    Men are naturally better suited for technical roles due to their analytical thinking.
    Women might struggle with the complex problem-solving required in software development.
    
    Preferred candidates:
    - Strong technical background (preferably male candidates)
    - Ability to work long hours without family distractions
    - Native English speakers only
    
    Contact: hiring@company.com
    """
    
    # Document 3: Content with privacy violations
    privacy_content = """
    Customer Database Export
    
    Customer Records:
    1. John Smith - SSN: 123-45-6789, Email: john.smith@email.com, Phone: 555-123-4567
    2. Sarah Johnson - SSN: 987-65-4321, Email: sarah.j@email.com, Phone: 555-987-6543
    3. Michael Brown - SSN: 456-78-9012, Email: m.brown@email.com, Phone: 555-456-7890
    
    Credit Card Information:
    - John Smith: 4532-1234-5678-9012, Exp: 12/25, CVV: 123
    - Sarah Johnson: 5678-9012-3456-7890, Exp: 08/26, CVV: 456
    
    Internal Use Only - Do Not Distribute
    """
    
    # Document 4: Content with harmful language
    harmful_content = """
    Community Guidelines Discussion
    
    Some users have been posting inappropriate content that violates our standards.
    These people are toxic and should be eliminated from our platform immediately.
    Their harmful behavior threatens the safety of our community.
    
    We need to take aggressive action against these individuals who spread hate.
    They deserve to be banned permanently without any chance of appeal.
    
    Moderator Notes:
    - Increase surveillance on suspicious accounts
    - Implement stricter punishment policies
    """
    
    documents = [
        ("clean_document.txt", clean_content, "Professional Policy Document"),
        ("biased_document.txt", biased_content, "Hiring Requirements Document"),
        ("privacy_document.txt", privacy_content, "Customer Data Export"),
        ("harmful_document.txt", harmful_content, "Community Guidelines Discussion")
    ]
    
    # Create temporary files
    temp_files = []
    for filename, content, description in documents:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(content)
        temp_file.close()
        
        temp_files.append({
            'path': Path(temp_file.name),
            'filename': filename,
            'description': description,
            'content_preview': content[:100] + "..." if len(content) > 100 else content
        })
    
    return temp_files


async def run_ethics_integration_demo():
    """Run the complete ethics integration demo"""
    console = Console()
    
    console.print("🛡️ [bold cyan]Ethics Framework + Document Processing Integration Demo[/bold cyan]\n")
    
    # Create test documents
    console.print("📝 Creating test documents...")
    test_files = await create_test_documents()
    
    console.print(f"✅ Created {len(test_files)} test documents\n")
    
    # Process each document with ethics checking
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        for file_info in test_files:
            task = progress.add_task(f"Processing {file_info['filename']}...", total=None)
            
            try:
                # Process document with integrated ethics checking
                result = await document_processor.process_document_with_ethics(
                    file_info['path']
                )
                
                result['file_info'] = file_info
                results.append(result)
                
                progress.update(task, description=f"✅ {file_info['filename']} processed")
                
            except Exception as e:
                console.print(f"❌ Error processing {file_info['filename']}: {e}")
                results.append({
                    'file_info': file_info,
                    'error': str(e),
                    'processing_result': None,
                    'ethics_report': None
                })
    
    console.print("\n📊 [bold yellow]Processing Results[/bold yellow]\n")
    
    # Display results for each document
    for result in results:
        file_info = result['file_info']
        ethics_report = result.get('ethics_report')
        processing_result = result.get('processing_result')
        
        # Determine panel color based on compliance
        if ethics_report:
            compliance_score = ethics_report.compliance_score
            if compliance_score >= 0.8:
                panel_color = "green"
                status_emoji = "✅"
            elif compliance_score >= 0.5:
                panel_color = "yellow"
                status_emoji = "⚠️"
            else:
                panel_color = "red"
                status_emoji = "❌"
        else:
            panel_color = "red"
            status_emoji = "❌"
            compliance_score = 0.0
        
        # Create panel content
        panel_content = f"{status_emoji} [bold]{file_info['description']}[/bold]\n"
        panel_content += f"File: {file_info['filename']}\n\n"
        
        if processing_result and processing_result.success:
            panel_content += f"📄 Document Processing: ✅ Success\n"
            panel_content += f"📝 Extracted Text: {len(processing_result.document.extracted_text)} characters\n"
        else:
            panel_content += f"📄 Document Processing: ❌ Failed\n"
        
        if ethics_report:
            panel_content += f"🛡️ Compliance Score: {compliance_score:.2f}/1.0\n"
            panel_content += f"⚠️ Issues Found: {len(ethics_report.issues_found)}\n"
            panel_content += f"🔍 Bias Detections: {len(ethics_report.bias_detections)}\n"
            panel_content += f"🔒 Privacy Violations: {len(ethics_report.privacy_violations)}\n"
        else:
            panel_content += f"🛡️ Ethics Analysis: ❌ Failed\n"
        
        console.print(Panel(
            panel_content,
            title=f"📋 {file_info['filename']}",
            border_style=panel_color
        ))
        
        # Show detailed issues if any
        if ethics_report and ethics_report.issues_found:
            issues_table = Table(show_header=True, header_style="bold magenta")
            issues_table.add_column("Category", style="cyan", width=15)
            issues_table.add_column("Severity", style="yellow", width=10)
            issues_table.add_column("Description", style="white", width=40)
            issues_table.add_column("Confidence", style="green", width=10)
            
            for issue in ethics_report.issues_found[:3]:  # Show first 3 issues
                issues_table.add_row(
                    issue.category.value.replace('_', ' ').title(),
                    issue.severity.value.upper(),
                    issue.description[:37] + "..." if len(issue.description) > 40 else issue.description,
                    f"{issue.confidence:.2f}"
                )
            
            console.print(issues_table)
            
            if len(ethics_report.issues_found) > 3:
                console.print(f"... and {len(ethics_report.issues_found) - 3} more issues")
        
        # Show recommendations
        if ethics_report and ethics_report.recommendations:
            console.print("💡 [bold blue]Recommendations:[/bold blue]")
            for i, rec in enumerate(ethics_report.recommendations[:2], 1):
                console.print(f"  {i}. {rec}")
            if len(ethics_report.recommendations) > 2:
                console.print(f"  ... and {len(ethics_report.recommendations) - 2} more")
        
        console.print()
    
    # Generate summary statistics
    console.print("📈 [bold yellow]Summary Statistics[/bold yellow]")
    
    total_docs = len(results)
    successful_processing = sum(1 for r in results if r.get('processing_result') and r['processing_result'].success)
    successful_ethics = sum(1 for r in results if r.get('ethics_report') is not None)
    
    compliant_docs = sum(1 for r in results 
                        if r.get('ethics_report') and r['ethics_report'].compliance_score >= 0.7)
    
    total_issues = sum(len(r['ethics_report'].issues_found) for r in results 
                      if r.get('ethics_report'))
    
    summary_table = Table(show_header=False)
    summary_table.add_column("Metric", style="cyan", width=25)
    summary_table.add_column("Value", style="green", width=15)
    summary_table.add_column("Percentage", style="yellow", width=15)
    
    summary_table.add_row("Total Documents", str(total_docs), "100%")
    summary_table.add_row("Successful Processing", str(successful_processing), f"{successful_processing/total_docs:.1%}")
    summary_table.add_row("Ethics Analysis Complete", str(successful_ethics), f"{successful_ethics/total_docs:.1%}")
    summary_table.add_row("Compliant Documents", str(compliant_docs), f"{compliant_docs/total_docs:.1%}")
    summary_table.add_row("Total Issues Found", str(total_issues), "-")
    
    console.print(summary_table)
    
    # Cleanup temporary files
    console.print("\n🧹 Cleaning up temporary files...")
    for file_info in test_files:
        try:
            file_info['path'].unlink()
        except Exception as e:
            console.print(f"Warning: Could not delete {file_info['path']}: {e}")
    
    console.print("\n✨ [green]Ethics integration demo complete![/green]")
    console.print("\n🎯 [bold]Key Features Demonstrated:[/bold]")
    console.print("  • Document processing with ethics compliance checking")
    console.print("  • Bias detection in document content")
    console.print("  • Privacy violation identification")
    console.print("  • Harmful content detection")
    console.print("  • Compliance scoring and recommendations")
    console.print("  • Integrated reporting and analytics")


if __name__ == "__main__":
    asyncio.run(run_ethics_integration_demo())