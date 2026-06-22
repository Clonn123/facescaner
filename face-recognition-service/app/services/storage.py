import uuid
import numpy as np
from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from datetime import datetime

from app.models.database_models import Employee, LivenessLog
from app.models.schemas import EmployeeResponse


class StorageService:
    """Сервис хранения данных сотрудников."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def register_employee(self, account_id: str, name: Optional[str] = None,
                               description: Optional[str] = None) -> Employee:
        """
        Регистрация или обновление сотрудника.
        Если сотрудник уже существует — возвращает его.
        """
        # Проверяем, существует ли сотрудник
        stmt = select(Employee).where(Employee.account_id == account_id)
        result = await self.db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        if employee:
            # Обновляем информацию если нужно
            if name:
                employee.name = name
            if description:
                employee.description = description
            await self.db.commit()
            await self.db.refresh(employee)
            return employee
        
        # Создаём нового сотрудника
        employee_id = str(uuid.uuid4())
        
        employee = Employee(
            employee_id=employee_id,
            account_id=account_id,
            name=name,
            description=description,
            embedding=[0.0] * 512,  # Инициализируем нулями
            faces_registered=0
        )
        
        self.db.add(employee)
        await self.db.commit()
        await self.db.refresh(employee)

        return employee

    async def update_embedding(self, employee_id: str, embedding: np.ndarray,
                               faces_count: int = 1) -> Optional[Employee]:
        """
        Обновление embedding сотрудника.
        
        Args:
            employee_id: ID сотрудника
            embedding: Массив embedding
            faces_count: Количество зарегистрированных лиц
            
        Returns:
            Обновлённый Employee или None
        """
        stmt = select(Employee).where(Employee.employee_id == employee_id)
        result = await self.db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        if not employee:
            return None
        
        # Сохраняем embedding как массив floats
        employee.embedding = embedding.tolist()
        employee.embedding_dim = len(embedding)
        employee.faces_registered = faces_count
        employee.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(employee)
        
        return employee

    async def add_face_to_employee(self, employee_id: str, 
                                   new_embedding: np.ndarray) -> Optional[Employee]:
        """
        Добавление нового face embedding к существующему сотруднику.
        Усредняет новый embedding с существующим.
        
        Args:
            employee_id: ID сотрудника
            new_embedding: Новый embedding
            
        Returns:
            Обновлённый Employee или None
        """
        stmt = select(Employee).where(Employee.employee_id == employee_id)
        result = await self.db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        if not employee or not employee.embedding:
            return None
        
        # Получаем существующий embedding как массив
        existing_emb = np.array(employee.embedding, dtype=np.float32)
        
        # Усредняем два embedding
        averaged = (existing_emb + new_embedding) / 2.0
        
        employee.embedding = averaged.tolist()
        employee.faces_registered += 1
        employee.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(employee)
        
        return employee

    async def get_all_employees_for_recognition(self) -> List[Tuple[str, np.ndarray]]:
        """
        Получение всех сотрудников для распознавания.
        
        Returns:
            Список кортежей (employee_id, embedding)
        """
        stmt = select(Employee.employee_id, Employee.embedding).where(
            Employee.embedding.isnot(None)
        )
        result = await self.db.execute(stmt)
        rows = result.all()
        
        candidates = []
        for employee_id, embedding_list in rows:
            if embedding_list:
                embedding = np.array(embedding_list)
                candidates.append((employee_id, embedding))
        
        return candidates

    async def get_employee(self, employee_id: str) -> Optional[EmployeeResponse]:
        """Получение информации о сотруднике."""
        stmt = select(Employee).where(Employee.employee_id == employee_id)
        result = await self.db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        if not employee:
            return None
        
        return EmployeeResponse(
            employee_id=employee.employee_id,
            account_id=employee.account_id,
            name=employee.name,
            description=employee.description,
            faces_registered=employee.faces_registered,
            created_at=employee.created_at,
            updated_at=employee.updated_at
        )

    async def get_employee_by_account_id(self, account_id: str) -> Optional[Employee]:
        """Получение сотрудника по account_id."""
        stmt = select(Employee).where(Employee.account_id == account_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_employee(self, employee_id: str) -> bool:
        """Удаление сотрудника."""
        stmt = select(Employee).where(Employee.employee_id == employee_id)
        result = await self.db.execute(stmt)
        employee = result.scalar_one_or_none()
        
        if not employee:
            return False
        
        await self.db.delete(employee)
        await self.db.commit()
        return True

    async def log_liveness(self, employee_id: Optional[str], account_id: Optional[str],
                          liveness_score: float, is_real: bool,
                          source: Optional[str] = None, ip_address: Optional[str] = None):
        """Логирование проверки живого присутствия."""
        log = LivenessLog(
            employee_id=employee_id,
            account_id=account_id,
            liveness_score=liveness_score,
            is_real=1 if is_real else 0,
            source=source,
            ip_address=ip_address
        )
        
        self.db.add(log)
        await self.db.commit()
