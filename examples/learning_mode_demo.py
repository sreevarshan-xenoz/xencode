#!/usr/bin/env python3
"""
Learning Mode Demo

Demonstrates the Learning Mode feature with CLI and TUI integration.
"""

import asyncio
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from xencode.features import FeatureConfig
from xencode.features.learning_mode import LearningModeFeature

console = Console()


async def demo_learning_mode():
    """Demonstrate Learning Mode functionality"""
    
    console.print(Panel.fit(
        "[bold cyan]Learning Mode Demo[/bold cyan]\n"
        "Interactive tutorials with adaptive difficulty",
        border_style="cyan"
    ))
    
    # Create feature config
    config = FeatureConfig(
        name='learning_mode',
        enabled=True,
        config={
            'enabled': True,
            'default_difficulty': 'beginner',
            'adaptive_enabled': True,
            'exercise_count': 5,
            'mastery_threshold': 0.8,
            'topics': ['python', 'javascript', 'rust', 'docker', 'git']
        }
    )
    
    # Initialize feature
    console.print("\n[yellow]Initializing Learning Mode...[/yellow]")
    feature = LearningModeFeature(config)
    await feature.initialize()
    console.print("[green]✓ Learning Mode initialized[/green]")
    
    # 1. List available topics
    console.print("\n[bold]1. Available Topics[/bold]")
    topics = await feature.get_topics()
    
    table = Table(title="📚 Learning Topics")
    table.add_column("Topic", style="cyan")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Time", style="blue")
    table.add_column("Description", style="white")
    
    for topic in topics:
        table.add_row(
            topic['id'],
            topic['difficulty'],
            f"{topic['estimated_time']}m",
            topic['description'][:40] + "..."
        )
    
    console.print(table)
    
    # 2. Start a topic
    console.print("\n[bold]2. Starting Python Tutorial[/bold]")
    result = await feature.start_topic('python')
    
    console.print(f"\n[cyan]Topic:[/cyan] {result['topic']['name']}")
    console.print(f"[cyan]Difficulty:[/cyan] {result['difficulty']}")
    console.print(f"[cyan]Estimated Time:[/cyan] {result['topic']['estimated_time']} minutes")
    
    lesson = result['lesson']
    console.print(f"\n[green]Lesson:[/green] {lesson.get('title', 'Introduction')}")
    console.print(Panel(lesson.get('content', ''), border_style="green"))
    
    if lesson.get('key_concepts'):
        console.print("\n[cyan]Key Concepts:[/cyan]")
        for concept in lesson['key_concepts']:
            console.print(f"  • {concept}")
    
    # 3. Get exercises
    console.print("\n[bold]3. Getting Exercises[/bold]")
    exercises = await feature.get_exercises('python', count=3)
    
    if exercises:
        console.print(f"\n[green]Found {len(exercises)} exercises:[/green]")
        for i, exercise in enumerate(exercises, 1):
            console.print(f"\n[cyan]{i}. {exercise['title']}[/cyan]")
            console.print(f"   {exercise['description']}")
            console.print(f"   Difficulty: {exercise['difficulty']}")
    else:
        console.print("[yellow]No exercises available for this topic yet[/yellow]")
    
    # 4. Simulate progress
    console.print("\n[bold]4. Simulating Learning Progress[/bold]")
    
    # Record some exercises
    console.print("\n[yellow]Recording exercise completions...[/yellow]")
    await feature.progress_tracker.record_exercise('python', passed=True, time_spent=5)
    await feature.progress_tracker.record_exercise('python', passed=True, time_spent=7)
    await feature.progress_tracker.record_exercise('python', passed=False, time_spent=10)
    await feature.progress_tracker.record_exercise('python', passed=True, time_spent=6)
    console.print("[green]✓ Recorded 4 exercise attempts[/green]")
    
    # 5. Check progress
    console.print("\n[bold]5. Checking Progress[/bold]")
    progress = await feature.get_progress('python')
    
    if progress:
        console.print(f"\n[cyan]Progress for Python:[/cyan]")
        console.print(f"  • Mastery Level: {progress['mastery_level']}")
        console.print(f"  • Exercises Completed: {progress['exercises_completed']}/{progress['exercises_total']}")
        console.print(f"  • Accuracy: {progress['accuracy']*100:.1f}%")
        console.print(f"  • Time Spent: {progress['time_spent']} minutes")
    
    # 6. Check mastery level
    console.print("\n[bold]6. Checking Mastery Level[/bold]")
    mastery = await feature.get_mastery_level('python')
    
    console.print(f"\n[cyan]Mastery Level:[/cyan]")
    console.print(f"  • Level: {mastery['mastery_level']}")
    console.print(f"  • Mastery: {mastery['mastery_percentage']:.1f}%")
    console.print(f"  • Accuracy: {mastery['accuracy']:.1f}%")
    
    # Provide feedback
    mastery_pct = mastery['mastery_percentage']
    if mastery_pct >= 90:
        console.print("\n[bold green]🏆 Excellent! You've mastered this topic![/bold green]")
    elif mastery_pct >= 70:
        console.print("\n[bold yellow]👍 Great progress! Keep practicing.[/bold yellow]")
    elif mastery_pct >= 50:
        console.print("\n[bold blue]📚 Good start! Continue learning.[/bold blue]")
    else:
        console.print("\n[bold cyan]🌱 Just getting started! Keep going![/bold cyan]")
    
    # 7. Get next recommended topic
    console.print("\n[bold]7. Getting Next Recommended Topic[/bold]")
    next_topic = await feature.get_next_topic()
    
    if next_topic:
        console.print(f"\n[green]Recommended next topic:[/green] {next_topic['name']}")
        console.print(f"[dim]{next_topic['description']}[/dim]")
    else:
        console.print("\n[yellow]No recommendations yet. Complete more topics![/yellow]")
    
    # 8. Overall progress
    console.print("\n[bold]8. Overall Learning Progress[/bold]")
    
    # Add progress for another topic
    await feature.progress_tracker.record_exercise('javascript', passed=True, time_spent=8)
    await feature.progress_tracker.record_exercise('javascript', passed=True, time_spent=9)
    
    overall_progress = await feature.get_progress()
    
    if overall_progress and overall_progress.get('topics'):
        table = Table(title="📊 Overall Progress")
        table.add_column("Topic", style="cyan")
        table.add_column("Mastery", style="green")
        table.add_column("Exercises", style="yellow")
        table.add_column("Accuracy", style="blue")
        
        for topic_progress in overall_progress['topics']:
            completion = f"{topic_progress['exercises_completed']}/{topic_progress['exercises_total']}"
            accuracy = f"{topic_progress['accuracy']*100:.1f}%"
            
            table.add_row(
                topic_progress['topic_id'],
                topic_progress['mastery_level'],
                completion,
                accuracy
            )
        
        console.print(table)
        
        console.print(f"\n[bold]Overall Statistics:[/bold]")
        console.print(f"  • Overall Mastery: {overall_progress['overall_mastery']*100:.1f}%")
        console.print(f"  • Total Time: {overall_progress['total_time']} minutes")
        console.print(f"  • Topics Started: {len(overall_progress['topics'])}")
    
    # Cleanup
    await feature.shutdown()
    
    # CLI Usage Examples
    console.print("\n" + "="*60)
    console.print(Panel.fit(
        "[bold cyan]CLI Usage Examples[/bold cyan]\n\n"
        "[yellow]Start learning a topic:[/yellow]\n"
        "  xencode learn start python\n"
        "  xencode learn start rust --difficulty intermediate\n\n"
        "[yellow]Check progress:[/yellow]\n"
        "  xencode learn progress\n"
        "  xencode learn progress --topic python\n\n"
        "[yellow]List topics:[/yellow]\n"
        "  xencode learn topics\n\n"
        "[yellow]Get exercises:[/yellow]\n"
        "  xencode learn exercises python\n"
        "  xencode learn exercises rust --count 10\n\n"
        "[yellow]Check mastery:[/yellow]\n"
        "  xencode learn mastery python",
        border_style="cyan"
    ))
    
    console.print("\n[green]✅ Demo completed successfully![/green]")


if __name__ == "__main__":
    asyncio.run(demo_learning_mode())
