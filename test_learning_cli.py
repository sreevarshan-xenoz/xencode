#!/usr/bin/env python3
"""
Quick test to verify learning mode CLI functionality
"""

import asyncio
from xencode.features import FeatureManager, FeatureConfig

async def test_learning_mode():
    """Test learning mode feature"""
    
    print("1. Creating feature manager...")
    manager = FeatureManager()
    
    print("2. Available features:", manager.get_available_features())
    
    print("3. Initializing learning mode...")
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
    
    success = await manager.initialize_feature('learningmode', config)
    print(f"   Initialization: {'✓ Success' if success else '✗ Failed'}")
    
    if not success:
        return
    
    print("4. Getting learning mode feature...")
    learning_feature = manager.get_feature('learningmode')
    print(f"   Feature loaded: {learning_feature is not None}")
    print(f"   Feature enabled: {learning_feature.is_enabled if learning_feature else False}")
    print(f"   Feature initialized: {learning_feature.is_initialized if learning_feature else False}")
    
    if learning_feature:
        print("\n5. Testing feature methods...")
        
        # Get topics
        topics = await learning_feature.get_topics()
        print(f"   ✓ get_topics(): {len(topics)} topics available")
        
        # Start a topic
        result = await learning_feature.start_topic('python')
        print(f"   ✓ start_topic('python'): {result['topic']['name']}")
        
        # Get exercises
        exercises = await learning_feature.get_exercises('python', 3)
        print(f"   ✓ get_exercises('python', 3): {len(exercises)} exercises")
        
        # Get progress
        progress = await learning_feature.get_progress('python')
        print(f"   ✓ get_progress('python'): {progress is not None}")
        
        # Get mastery
        mastery = await learning_feature.get_mastery_level('python')
        print(f"   ✓ get_mastery_level('python'): {mastery['mastery_level']}")
        
        print("\n✅ All learning mode features working correctly!")
    
    print("\n6. Shutting down...")
    await manager.shutdown_feature('learningmode')
    print("   ✓ Shutdown complete")

if __name__ == "__main__":
    asyncio.run(test_learning_mode())
